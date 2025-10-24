# aioresonate

[![pypi_badge](https://img.shields.io/pypi/v/aioresonate.svg)](https://pypi.python.org/pypi/aioresonate)

Async Python library implementing the [Resonate Protocol](https://github.com/Resonate-Protocol/spec).

For a WIP reference implementation of a server using this library, see [Music Assistant](https://github.com/music-assistant/server/tree/resonate/music_assistant/providers/resonate)

[![A project from the Open Home Foundation](https://www.openhomefoundation.org/badges/ohf-project.png)](https://www.openhomefoundation.org/)

## CLI Client

> **Note:** The CLI client will be moved to a separate repository in the future.

This repository also includes a highly experimental CLI client for testing and development purposes.
You can install and run it by:

1. Cloning this repository:
```bash
git clone https://github.com/Resonate-Protocol/aioresonate.git
cd aioresonate
```

2. Installing the package with CLI dependencies:
```
pip install --user ".[cli]"
```
(The `.[cli]` installs the `resonate-cli` command)

3. Running the CLI:
```
resonate-cli
```

4. Uninstalling the package:
```
pip uninstall aioresonate
```

The CLI client will automatically connect to a Resonate server on your local network and be available for playback.

### Client Identification

If you want to run multiple CLI clients simultaneously, each must have a unique identifier:

```bash
resonate-cli --id my-client-1 --name "Kitchen"
resonate-cli --id my-client-2 --name "Bedroom"
```

- `--id`: A unique identifier for this client (required if running multiple instances)
- `--name`: A friendly name displayed on the server (optional)

### Limitations & Known Issues

This client is highly experimental and has many limitations:

- Does not recover when it loses connection to the server
- Slowly drifts out of sync over time
- Only tested on Linux
