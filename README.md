# aioresonate

[![pypi_badge](https://img.shields.io/pypi/v/aioresonate.svg)](https://pypi.python.org/pypi/aioresonate)

Async Python library implementing the [Resonate Protocol](https://github.com/Resonate-Protocol/spec).

For a WIP reference implementation of a server using this library, see [Music Assistant](https://github.com/music-assistant/server/tree/resonate/music_assistant/providers/resonate)

[![A project from the Open Home Foundation](https://www.openhomefoundation.org/badges/ohf-project.png)](https://www.openhomefoundation.org/)

## CLI Client

> **Note:** The CLI client is currently included in the `aioresonate` library for development purposes. Once the Resonate Protocol stabilizes, it will be moved to a separate repository and package. This will require users to uninstall `aioresonate[cli]` and install the new CLI package separately.

This repository includes a highly experimental CLI client for testing and development purposes.

### Quick Start

**Run directly with [uv](https://docs.astral.sh/uv/getting-started/installation/):**
```bash
uvx --from "aioresonate[cli]" resonate-cli
```

### Installation

**With pip:**
```bash
pip install "aioresonate[cli]"
```

**With uv:**
```bash
uv tool install "aioresonate[cli]"
```

<details>
<summary>Install from source</summary>

```bash
git clone https://github.com/Resonate-Protocol/aioresonate.git
cd aioresonate
pip install ".[cli]"
```

</details>

**After installation, run:**
```bash
resonate-cli
```

The CLI client will automatically connect to a Resonate server on your local network and be available for playback.

### Configuration Options

#### Client Identification

If you want to run multiple CLI clients on the **same computer**, you can specify unique identifiers:

```bash
resonate-cli --id my-client-1 --name "Kitchen"
resonate-cli --id my-client-2 --name "Bedroom"
```

- `--id`: A unique identifier for this client (optional; defaults to `resonate-cli-<hostname>`, useful for running multiple instances on one computer)
- `--name`: A friendly name displayed on the server (optional; defaults to hostname)

#### Audio Output Device Selection

By default, the CLI client uses your system's default audio output device. You can list available devices or select a specific device:

**List available audio devices:**
```bash
resonate-cli --list-audio-devices
```

This displays all audio output devices with their IDs, channel configurations, and sample rates. The default device is marked.

**Select a specific audio device:**
```bash
resonate-cli --audio-device 2
```

This is particularly useful for headless devices or when you want to route audio to a specific output.

#### Adjusting Playback Delay

The CLI supports adjusting playback delay to compensate for audio hardware latency or achieve better synchronization across devices.

**Setting delay at startup:**
```bash
resonate-cli --static-delay-ms -100
```

> **Note:** Based on limited testing, the delay value is typically a negative number (e.g., `-100` or `-150`) to compensate for audio hardware buffering.

**Adjusting delay in real-time:**
While the client is running, you can use the `delay` command:
- `delay` - Show current delay value
- `delay <ms>` - Set absolute delay (e.g., `delay -100`)
- `delay + <ms>` - Increase delay (e.g., `delay + 50`)
- `delay - <ms>` - Decrease delay (e.g., `delay - 25`)

The synchronization will seamlessly adjust to the new delay value within a couple of seconds.

#### Debugging & Troubleshooting

If you experience synchronization issues or audio glitches, you can enable detailed logging to help diagnose the problem:

```bash
resonate-cli --log-level DEBUG
```

This provides detailed information about time synchronization. The output can be helpful when reporting issues.

### Limitations & Known Issues

This client is highly experimental and has several known limitations:

- **Platform Support**: Only tested on Linux; macOS and Windows support untested
- **Format Support**: Currently fixed to uncompressed 44.1kHz 16-bit stereo PCM
- **CLI User Experience**: The CLI is pretty bare bones for now
- **Configuration Persistence**: Settings are not persistently stored; delay must be reconfigured on each restart using the `--static-delay-ms` option
