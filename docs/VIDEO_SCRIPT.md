# Video script (2 minutes at most)

Screen recording with voice-over. Record the browser at https://rl8596-pneumoshift.static.hf.space/ in a window
about 1,400 pixels wide, so both model cards are visible. Open the page once before recording, so the two models
are already downloaded. Speak slowly: the text below is about 270 words, which fits 2 minutes.

Numbers marked **[v2]** depend on the second generation of models. Until those results are in the README, use the
sentence in brackets instead.

| Time | On screen | Say |
|---|---|---|
| 0:00 to 0:15 | The page, empty. Then the line "For a clinic without a radiologist". | "A small clinic takes a chest X-ray, but nobody there can read it. In Malawi there are three radiologists for almost twenty million people. Our tool gives a second look: which X-ray should a doctor see first?" |
| 0:15 to 0:30 | Click the example "Child, pneumonia". Both cards turn red: "Flag for priority review". Point at the heatmaps. | "The usual tutorial trains on one public dataset. It holds only children aged one to five, from one hospital. On such children the model is right 94 times in 100. Here it flags the X-ray and shows where it looked." |
| 0:30 to 0:55 | Click "Adult (46), normal". Left card: score 90%, orange warning "Do not rely on this result". Right card: green, 5%. | "But what happens with an adult from another hospital? This is a healthy 46-year-old. The children-only model gives a score of 90 percent. Our tool notices: this X-ray looks unlike anything the model learned from, do not rely on it. The second model also learned from adults. It says 5 percent, no sign of pneumonia." |
| 0:55 to 1:15 | Click "Not an X-ray (a letter)". Both cards show the warning. | "The model itself cannot say 'this is not an X-ray'. It always gives a score. The warning catches that too. And the picture never leaves your device: both models run in the browser." |
| 1:15 to 1:45 | README results table, then the figure `shift_kids_only_subgroups.png`, then `heatmap_pointing_test.png`. | "We tested on 7,694 adults the models never saw. The children-only model drops from 0.98 to 0.82. Adult X-rays repair part of that: 0.87 **[v2: and with all adult X-rays, …]**. We also compared against simple baselines and tested the heatmaps against radiologists' markings: always pointing at the most common spot is right more often than our heatmaps. So we say in the demo that a heatmap is not a map of the disease." |
| 1:45 to 2:00 | The "Limits you should know about" panel. | "Our limits: two hospitals, public data, and the adult labels are radiologists' findings, not confirmed diagnoses. Adults who are sick with something else are often flagged too. Next we would test on a third hospital. This is a research demo, not a medical device." |

## Before recording

- Check each click once: the three examples must show what the script says. If a number on screen differs from the
  script, say what is on screen.
- The examples in the demo were drawn at random from the test X-rays, they were not hand-picked. You may say so.
- No names, no school, no faces: judging is blind.
- Upload unlisted to YouTube, or to Google Drive with "anyone with the link can view". Open the link in a
  logged-out browser window.
