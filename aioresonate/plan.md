# Plan

- Channel will be an internal concept of a audio stream with different data.
    - A stream for a client will receive (sometimes resampled and encoded) audio data from a channel
    - Default channel must always be provided. This will be used for:
        - All clients when no special DSP is required
        - Clients that have no DSP if DSP is only used on a subset of clients
        - Generating additional metadata (like FFT or maybe beat timings) for the future visualizer role
    - Additional channels can be passed by MediaStream (instead of the generator by itself) play_media (replaces DirectStreamSession)
    - MediaStream has a generator for the main channel, and optionally methods similar to DirectStreamSession for manual insertion
- ResonateGroup will be responsible managing the resonate stream, its task will be:
    - Sending stream start/stop messages, with player payloads passed from Streamer (like codec)
    - Handling requests for format changes
    - Configuring the Streamer with the current topology
    - Running the Streamer, including collecting and passing raw audio chunks from all channels to the Streamer
- Streamer (in stream.py) will adapt the incoming channels to the requirements of the players
    - Streamer has multiple responsibilities that run single threaded for the whole group (blocking functions run by ResonateGroup):
        - configure: update the internal state of objects (like queues and helper dicts) based on passed in topology description
            - includes a list for client and their preferred format, chunk size, buffer size, and channel
            - and list of channel byte formats
            - returns session update/start payloads (only for those where it changed)
        - prepare: receives audio data from all channels (raw bytes, asserts that all are the same size! ResonateGroup is responsible for that)
            1. runs all required resampling
            2. Puts resampled audio into a buffer for later chunking
            3. Creates (no copies if possible) chunks for each chunk size (chunk size based on the codec, so every codec runs optimally)
            4. Encodes chunks (with the correct sample rates and sizes) and saves them for later so encoding is deduplicated for similar players
        - send: creates messages for players and queues them for sending
            - Distributes prepared audio chunks to each player
            - Returns True if player buffers are not fully exhausted yet, meaning that prepare and another call to send should immediately be done

After implementing this plan, we should be able to:
- deduplicate as much effort as possible between: resampling, chunking, and encoding
- Support individual device DSP without duplicated code
