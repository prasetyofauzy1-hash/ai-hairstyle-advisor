# Fixed prompt for the blinded vision pass

Classify the face shape in each numbered tile. Use the visible face outline:
face length versus cheekbone width, forehead width at the temples, jaw width
and angle, and how quickly the lower face narrows toward the chin. Ignore
hairstyle and hair volume, ears, facial hair outside the jaw, background, and
any notions of who the person might be. Estimate the underlying outline when
pose, expression, hair, or framing makes the boundary less visible.

- **Diamond:** cheekbones are distinctly widest; both the forehead and jaw are narrower, with a tapered chin.
- **Heart:** the forehead and temples are relatively broad against the narrow jaw and tapered chin.
- **Oblong:** length clearly dominates width and the sides are fairly parallel or similarly wide across forehead, cheeks, and jaw.
- **Oval:** moderately longer than wide, with softly curved sides and gradual taper into a rounded jaw/chin.
- **Round:** length and width are similar, cheeks broad, and the jaw and chin have a soft arc.
- **Square:** forehead and jaw widths are similar, with an angular jaw and comparatively broad/squared chin.

Choose exactly one label per tile, even if the image is ambiguous. For
Diamond versus Heart, compare temple width to cheekbone width. For Square
versus Oblong, prioritize clear elongation; for Round versus Oval, prioritize
the length-to-width ratio. Never consult source labels, identities, cohort
membership, or other classifiers. Do not use examples or feedback to revise
the rubric. Record tile IDs and predictions before joining any ground truth.
