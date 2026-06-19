# Chapter 4 Notes

Updated: 2026-06-20 03:23:28 Asia/Shanghai

Current screening status:

- Four-class structure candidate pool is complete.
- Road-surface domain consistency filter is complete.
- All retained class counts are >= 50.
- Next step: LPIPS perceptual diversity selection for final Ours-200 set.


## Ours-200 Final Selection

Updated: 2026-06-20 03:38:56 Asia/Shanghai

The final Ours-200 set was generated using domain-constrained LPIPS diversity selection.

- D00=50
- D10=50
- D20=50
- D40=50
- Backbone: `lpips_alex`
- Ready for YOLO dataset preparation: True
