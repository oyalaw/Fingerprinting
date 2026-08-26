# v0.8.8 Interactive Operating-System Selection

## What changed

The interactive client/server configuration no longer asks the user to type a
routine operating-system label. The workflow is now:

1. Select the hardware device from a numbered menu.
2. Select the operating system from a numbered menu tailored to that device.
3. Use `custom` only when the required OS is not in the catalogue.

Example for a Dell desktop:

```text
Select device label
  1. jetson_agx_orin
  ...
  4. dell_desktop
Selection: 4

Select operating system
  1. ubuntu
  2. windows_11
  3. windows_10
  4. debian
  5. fedora
  6. other_linux
  7. custom
Selection: 1
```

The resulting metadata is unambiguous:

```yaml
device:
  label: dell_desktop
  operating_system: ubuntu
```

## Device-aware choices

- Jetson devices: Ubuntu, other Linux, custom.
- Dell devices: Ubuntu, Windows 11, Windows 10, Debian, Fedora, other Linux, custom.
- Generic Linux desktop: common Linux distributions plus custom.
- Generic Windows desktop: Windows 11, Windows 10, custom.
- Raspberry Pi 5: Raspberry Pi OS, Ubuntu, other Linux, custom.
- Android phone: Android, custom.
- iPhone: iOS, custom.
- iPad: iPadOS, iOS, custom.
- MacBook: macOS, Ubuntu, other Linux, custom.

This change is metadata-only; it does not change the attacker feature set.
Operating-system and device labels remain ground truth/system-characterization
fields and are excluded from proxy predictor features.
