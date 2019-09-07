## 0.1.2 (2019/09/07)
- Split in two crates: diffr-lib contains reusable parts, while diffr
  only contains application logic.
  
- Fix a bug in display code that messed up the colors in diffs with
  lines starting with dashes.
  
- Configuration: default to use 16 colors everywhere (Github #16).

## 0.1.1 (2019/07/15)
- Add --colors flag to customize faces propertized by diffr (Github #3).
  This changes the default colors used on linux and macOS.
  The default still works on windows.

## 0.1.0 (2019/07/01) Initial release.
- Initial release.
