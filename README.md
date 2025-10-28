# aioresonate

[![pypi_badge](https://img.shields.io/pypi/v/aioresonate.svg)](https://pypi.python.org/pypi/aioresonate)

Async Python library implementing the [Resonate Protocol](https://github.com/Resonate-Protocol/spec).

For a WIP reference implementation of a server using this library, see [Music Assistant](https://github.com/music-assistant/server/tree/resonate/music_assistant/providers/resonate)

[![A project from the Open Home Foundation](https://www.openhomefoundation.org/badges/ohf-project.png)](https://www.openhomefoundation.org/)

## CLI Client

> **Note:** The CLI client is currently included in the `aioresonate` library for development purposes. Once the Resonate Protocol stabilizes, it will be moved to a separate repository and package. This will require users to uninstall `aioresonate[cli]` and install the new CLI package separately.

This repository includes a highly experimental CLI client for testing and development purposes.

### Installation

Install from PyPI:
```bash
pip install "aioresonate[cli]"
```

<details>
<summary>Install from source</summary>

```bash
git clone https://github.com/Resonate-Protocol/aioresonate.git
cd aioresonate
pip install ".[cli]"
```

</details>

### Running the CLI

```bash
resonate-cli
```

The CLI client will automatically connect to a Resonate server on your local network and be available for playback.

### Configuration Options

#### Client Identification

If you want to run multiple CLI clients simultaneously, each must have a unique identifier:

```bash
resonate-cli --id my-client-1 --name "Kitchen"
resonate-cli --id my-client-2 --name "Bedroom"
```

- `--id`: A unique identifier for this client (required if running multiple instances)
- `--name`: A friendly name displayed on the server (optional)

#### Adjusting Playback Delay

The CLI supports adjusting playback delay to compensate for audio hardware latency or achieve better synchronization across devices.

**Setting delay at startup:**
```bash
resonate-cli --static-delay-ms 150
```

**Adjusting delay in real-time:**
While the client is running, you can use the `delay` command:
- `delay` - Show current delay value
- `delay <ms>` - Set absolute delay (e.g., `delay 200`)
- `delay + <ms>` - Increase delay (e.g., `delay + 50`)
- `delay - <ms>` - Decrease delay (e.g., `delay - 25`)

Changing the delay clears the audio buffer to prevent desynchronization.

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
- **Playback Start Glitches**: When starting playback, audio dropouts or pitch shifts may occur during initial synchronization
- **Performance Requirements**: Low-powered devices may not keep up. While total CPU usage is very minimal, it currently requires low latency. This will be optimized in a future version
- **CLI User Experience**: The CLI is pretty bare bones for now
- **Configuration Persistence**: Settings are not persistently stored; delay must be reconfigured on each restart using the `--static-delay-ms` option
