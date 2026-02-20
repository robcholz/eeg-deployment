# Scripts

## Audio

```shell
python scripts/play_wav.py --device sysdefault --rate 48000 --channels 2 --frames-per-block 256 --wav <filename>
```

```shell
python scripts/alsa_record_and_play.py --device sysdefault --rate 48000 --channels 2 --frames-per-block 256 --seconds 2 --wav /tmp/alsa_record_and_play.wav
```

```shell
python scripts/alsa_loopback.py --device sysdefault --rate 48000 --channels 2 --frames-per-block 256 --queue-frames 64 --seconds 3
```
