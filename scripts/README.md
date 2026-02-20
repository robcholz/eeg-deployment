# Scripts

## Audio

```shell
python3 scripts/alsa_channel_test.py --device hw:2,0 --rate 48000 --frames-per-block 256
```

```shell
python3 scripts/alsa_record_and_play.py --device hw:2,0 --rate 48000 --channels 2 --frames-per-block 256 --seconds 2 --wav /tmp/alsa_record_and_play.wav
```

```shell
python3 scripts/alsa_loopback.py --device hw:2,0 --rate 48000 --channels 2 --frames-per-block 256 --queue-frames 64 --seconds 3
```