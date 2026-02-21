#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-time WebSocket server for CuBFF simulation visualization."""

import argparse
import asyncio
import json
import os
import queue
import struct
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) + "/bin/"
)
import cubff

try:
    import websockets
    from websockets.asyncio.server import serve
except ImportError:
    print("Error: 'websockets' package required. Install with: pip install websockets")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="CuBFF real-time visualizer server")
    p.add_argument("--lang", default="bff_noheads", help="Language to simulate")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--num", type=int, default=131072, help="Number of programs")
    p.add_argument("--port", type=int, default=8765, help="WebSocket port")
    p.add_argument(
        "--callback_interval", type=int, default=1, help="Epochs between updates"
    )
    p.add_argument(
        "--eval_selfrep", action="store_true", help="Enable self-replication detection"
    )
    p.add_argument("--mutation_prob", type=int, default=None, help="Mutation probability")
    p.add_argument(
        "--max_visible",
        type=int,
        default=1024,
        help="Max programs shown in heatmap (subsamples if needed)",
    )
    p.add_argument("--max_epochs", type=int, default=0, help="Stop after N epochs (0=infinite)")
    return p.parse_args()


class SimulationBroadcaster:
    """Runs simulation in a thread, broadcasts state over WebSocket."""

    def __init__(self, args):
        self.args = args
        self.language = cubff.GetLanguage(args.lang)
        self.data_queue = queue.Queue(maxsize=4)
        self.init_data = None
        self.init_event = threading.Event()
        self.stop_flag = threading.Event()
        self.clients = set()
        self.lock = asyncio.Lock()
        self._decimation = 1
        self._callback_count = 0

    def _make_init_message(self, state):
        byte_colors = [[c[0], c[1], c[2]] for c in state.byte_colors]
        return json.dumps(
            {
                "type": "init",
                "byte_colors": byte_colors,
                "num_programs": self.args.num,
                "tape_size": 64,
                "lang": self.args.lang,
                "eval_selfrep": self.args.eval_selfrep,
                "max_visible": self.args.max_visible,
                "callback_interval": self.args.callback_interval,
            }
        )

    def _make_metrics_message(self, state):
        freq = [[p[0], p[1]] for p in state.frequent_bytes]
        uncommon = [[p[0], p[1]] for p in state.uncommon_bytes]

        rep_count = 0
        if self.args.eval_selfrep and len(state.replication_per_prog) > 0:
            rep_count = sum(
                1 for r in state.replication_per_prog if r >= cubff.kSelfrepThreshold
            )

        # Compute byte histogram and positional entropy from full soup
        soup_array = np.frombuffer(bytes(state.soup), dtype=np.uint8).reshape(-1, 64)
        byte_histogram = (
            np.bincount(soup_array.ravel(), minlength=256) / soup_array.size
        ).tolist()

        positional_entropy = []
        for pos in range(64):
            counts = np.bincount(soup_array[:, pos], minlength=256)
            probs = counts[counts > 0] / soup_array.shape[0]
            positional_entropy.append(
                round(float(-np.sum(probs * np.log2(probs))), 4)
            )

        return json.dumps(
            {
                "type": "metrics",
                "epoch": state.epoch,
                "elapsed_s": state.elapsed_s,
                "mops_s": state.mops_s,
                "brotli_bpb": state.brotli_bpb,
                "h0": state.h0,
                "higher_entropy": state.higher_entropy,
                "brotli_size": state.brotli_size,
                "bytes_per_prog": state.bytes_per_prog,
                "total_ops": state.total_ops,
                "ops_per_run": state.ops_per_run,
                "frequent_bytes": freq,
                "uncommon_bytes": uncommon,
                "replicator_count": rep_count,
                "replicator_fraction": rep_count / self.args.num if self.args.num else 0,
                "byte_histogram": byte_histogram,
                "positional_entropy": positional_entropy,
            }
        )

    def _make_soup_message(self, state):
        soup_bytes = bytes(state.soup)
        tape_size = 64
        total_programs = len(soup_bytes) // tape_size
        max_vis = min(self.args.max_visible, total_programs)

        if max_vis < total_programs:
            stride = total_programs // max_vis
        else:
            stride = 1
            max_vis = total_programs

        # Build replicator scores for visible programs (0-13 actual scores)
        has_selfrep = self.args.eval_selfrep and len(state.replication_per_prog) > 0
        rep_flags = bytearray(max_vis)
        if has_selfrep:
            for i in range(max_vis):
                prog_idx = i * stride
                if prog_idx < len(state.replication_per_prog):
                    rep_flags[i] = min(int(state.replication_per_prog[prog_idx]), 255)

        # Header: type(1) + visible_count(4) + stride(4) + has_selfrep(1)
        header = struct.pack("<BIIB", 0x01, max_vis, stride, 1 if has_selfrep else 0)

        # Subsample soup data
        chunks = []
        for i in range(max_vis):
            start = i * stride * tape_size
            chunks.append(soup_bytes[start : start + tape_size])

        return header + b"".join(chunks) + (bytes(rep_flags) if has_selfrep else b"")

    def sim_callback(self, state):
        if self.stop_flag.is_set():
            return True

        if self.init_data is None:
            self.init_data = self._make_init_message(state)
            self.init_event.set()

        self._callback_count += 1
        if self._callback_count % self._decimation != 0:
            return False

        metrics_json = self._make_metrics_message(state)
        soup_bin = self._make_soup_message(state)

        try:
            self.data_queue.put_nowait((metrics_json, soup_bin))
        except queue.Full:
            # Drop oldest frame for backpressure
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.data_queue.put_nowait((metrics_json, soup_bin))
            except queue.Full:
                pass

        if self.args.max_epochs > 0 and state.epoch >= self.args.max_epochs:
            return True
        return False

    def run_simulation(self):
        params = cubff.SimulationParams()
        params.num_programs = self.args.num
        params.seed = self.args.seed
        params.callback_interval = self.args.callback_interval
        params.eval_selfrep = self.args.eval_selfrep
        if self.args.mutation_prob is not None:
            params.mutation_prob = self.args.mutation_prob
        print(f"[sim] Starting simulation: lang={self.args.lang}, seed={self.args.seed}, "
              f"programs={self.args.num}, eval_selfrep={self.args.eval_selfrep}")
        self.language.RunSimulation(params, None, self.sim_callback)
        print("[sim] Simulation finished.")

    async def register(self, ws):
        async with self.lock:
            self.clients.add(ws)
        # Send init data if available
        self.init_event.wait(timeout=30)
        if self.init_data:
            try:
                await ws.send(self.init_data)
            except websockets.ConnectionClosed:
                pass

    async def unregister(self, ws):
        async with self.lock:
            self.clients.discard(ws)

    async def handler(self, ws):
        await self.register(ws)
        try:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                    if data.get("type") == "set_decimation":
                        self._decimation = max(1, int(data["value"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
        except websockets.ConnectionClosed:
            pass
        finally:
            await self.unregister(ws)

    async def broadcast_loop(self):
        while not self.stop_flag.is_set():
            try:
                metrics_json, soup_bin = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.data_queue.get(timeout=0.5)
                )
            except queue.Empty:
                continue

            async with self.lock:
                clients = set(self.clients)

            if not clients:
                continue

            # Send metrics JSON then soup binary
            for ws in clients:
                try:
                    await ws.send(metrics_json)
                    await ws.send(soup_bin)
                except websockets.ConnectionClosed:
                    await self.unregister(ws)


def run_http_server(port, directory):
    handler = partial(SimpleHTTPRequestHandler, directory=directory)
    server = HTTPServer(("0.0.0.0", port), handler)
    server.serve_forever()


async def main():
    args = parse_args()
    broadcaster = SimulationBroadcaster(args)

    # Start HTTP file server for the dashboard
    http_port = args.port + 1
    html_dir = os.path.dirname(os.path.abspath(__file__))
    http_thread = threading.Thread(
        target=run_http_server, args=(http_port, html_dir), daemon=True
    )
    http_thread.start()

    # Start simulation thread
    sim_thread = threading.Thread(target=broadcaster.run_simulation, daemon=True)
    sim_thread.start()

    # Start WebSocket server
    print(f"[ws]   WebSocket server on ws://localhost:{args.port}")
    print(f"[http] Dashboard at http://localhost:{http_port}/realtime-visualizer.html")
    print()

    async with serve(broadcaster.handler, "0.0.0.0", args.port):
        await broadcaster.broadcast_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
