# Video script (2 minutes at most)

Screen recording with voice-over. Record the browser at https://rl8596-pneumoshift.static.hf.space/ in a window
about 1,400 pixels wide, so both model cards are visible. Open the page once before recording, so the two models
are already downloaded.

**Pace.** The text below is about 285 spoken words, numbers counted the way you say them. That is 142 words per
minute: brisk, but it fits. The version before 15 Nov had about 330 words and its results part needed 200 words
per minute, so it did not fit. Read it aloud once with a stopwatch. If you run over, cut the sentence about the
heatmaps in the fifth row first.

The numbers of the second generation of models (README section 6) are in since 14 Nov: 0.88 with all adult
X-rays against 0.87 with 6,000, not a shown improvement.
An extra check on 15 Nov is in README section 6: three networks answering together reach 0.886 (exploratory). It
is left out of the voice-over on purpose, to stay within 2 minutes. It did not change the demo, the two published
models or the verdict. At 1:10, show the table under "Results" near the top of the README: the extra check is
not in that table. If someone asks about it in the live round: "That is an extra check. It is exploratory and
not in the demo."

| Time | On screen | Say |
|---|---|---|
| 0:00 to 0:15 | The page, empty. Then the line "For a clinic without a radiologist". | "A small clinic takes a chest X-ray, but nobody there can read it. In Malawi there are three radiologists for almost twenty million people. Our tool gives a second look: which X-ray should a doctor see first?" |
| 0:15 to 0:30 | Click the example "Child, pneumonia". Both cards turn red: "Flag for priority review". Point at the heatmaps. | "The usual tutorial model learns from one public dataset: only children aged one to five, from one hospital. There it is right 94 times in 100. Here it flags the X-ray and shows where it looked." |
| 0:30 to 0:55 | Click "Adult (46), normal". Left card: score 90%, orange warning "Do not rely on this result". Right card: green, 5%. | "But what happens with an adult from another hospital? This is a healthy 46-year-old. The children-only model gives a score of 90 percent. Our tool notices: this X-ray looks unlike anything the model learned from, do not rely on it. The second model also learned from adults. It says 5 percent, no sign of pneumonia." |
| 0:55 to 1:10 | Click "Not an X-ray (a letter)". Both cards show the warning. | "The model itself cannot say 'this is not an X-ray'. It always gives a score. The warning catches that too. And the picture never leaves your device: both models run in the browser." |
| 1:10 to 1:45 | The README on GitHub: the table under "Results", then Figure 2 (by group, in section 3), then Figure 4 (heatmap test, in section 5). | "We tested on almost 7,700 adults that no model had seen, and against simple baselines. The children-only model's score falls from 0.98 to 0.82. Six thousand adult X-rays repair part of that: 0.87. All fifteen thousand give 0.88. By the rule we wrote down before training, that is not a shown improvement. We also tested the heatmaps against radiologists' markings. One fixed spot is right more often than our heatmaps." |
| 1:45 to 2:00 | The "Limits you should know about" panel. | "Our limits: two hospitals, and the adult labels are radiologists' findings, not confirmed diagnoses. Adults who are sick with something else are often flagged too. Next: a third hospital. A research demo, not a medical device." |

## Before recording

- Check each click once: the three examples must show what the script says. If a number on screen differs from the
  script, say what is on screen.
- The examples in the demo were drawn at random from the test X-rays, they were not hand-picked. You may say so.
- No names, no school, no faces: judging is blind.
- Upload unlisted to YouTube, or to Google Drive with "anyone with the link can view". Open the link in a
  logged-out browser window.

## Questions you may get in the live round

Short answers. Every number is in the README; say where.

- **"Your accuracy is below always answering no. Why show this?"** A screening tool is set to catch 95% of the
  cases, and pays with false alarms. 78% of the adults are non-cases, so "always no" is right for 78% and catches
  nobody. That is why we report cases caught and non-cases cleared, not accuracy alone (Results, first point).
- **"Is lung opacity the same as pneumonia?"** No. It is what a radiologist saw on the picture, not a confirmed
  diagnosis ("Limits" under "More results").
- **"Why did more adult X-rays not help?"** From 6,000 to 15,142 the score moved by +0.006, and the interval
  touches zero. What stays hard is telling lung opacity from other lung disease, and one threshold for two kinds
  of X-ray (section 6, section 3).
- **"What can the warning do, and what not?"** It notices an unfamiliar hospital: it fires for 81% of the adults
  and 7% of the children. It does not find the wrong answers (section 4).
- **"What about the 0.886?"** An extra check, exploratory: three networks together, run after we had seen the
  test results. It is not in the demo and does not change the verdict (section 6).
- **"What is your own work here?"** Section 10 lists what is ours and what is not. Section 11 says how we used an
  AI assistant. Answer this one in your own words.
