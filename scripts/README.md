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

You might want to make sure the headphones are phyiscally connected well.

## Volume Control

```shell
pactl list short sinks

pactl set-default-sink alsa_output.usb-Andrea_Electronics_Andrea_Comm_USB-SA_Headset_SEP_2015-00.analog-stereo

pactl set-sink-mute @DEFAULT_SINK@ 0
pactl set-sink-volume @DEFAULT_SINK@ 80%
```
