# aioresonate

[![pypi_badge](https://img.shields.io/pypi/v/aioresonate.svg)](https://pypi.python.org/pypi/aioresonate)

Async Python library implementing the [Resonate Protocol](https://github.com/Resonate-Protocol/spec).

For a WIP reference implementation of a server using this library, see [Music Assistant](https://github.com/music-assistant/server/tree/resonate/music_assistant/providers/resonate)

[![A project from the Open Home Foundation](https://www.openhomefoundation.org/badges/ohf-project.png)](https://www.openhomefoundation.org/)

## CLI Client

Install the optional CLI dependencies and run the bundled client:

```
pip install "aioresonate[cli]"
resonate-cli --url ws://localhost:1789/resonate
```

The CLI streams audio (requires `sounddevice`) and offers simple keyboard commands for playback and volume control. The CLI discovers Resonate servers via mDNS (service `_resonate-server._tcp.local.`) by default; override with `--url` to connect directly. Use `--static-delay-ms <ms>` to add extra playback latency if your device needs additional buffering, and type `delay` commands in the terminal to inspect or tweak the value while running.
