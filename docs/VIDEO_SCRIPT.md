# Video script (2 minutes at most)

The video is 1:59 long, 1920 x 1080, with captions and no sound. It follows the structure from the Project Briefs:
the problem (14 s), the demo (43 s), how it works (43 s), limits and next steps (19 s). The demo part
shows the real screen recording and screenshot of the demo from `docs/media/`. The X-rays are held-out test
X-rays from `app/examples/`, and the results on screen are read from the files in `results/`.

**Pace.** The captions are 290 spoken words, numbers counted the way you say them: 146 words per minute.
To add a voice, read each caption while it is on screen.

The extra check with three networks answering together (0.886, README section 6) is left out on purpose. It is
not in the table under "Results" either. If someone asks about it in the live round: "That is an extra check. It
is exploratory and not in the demo."

| Time | Part | On screen | Caption |
|---|---|---|---|
| 0:00 to 0:14 | The problem | Title and the line "A prototype for a clinic that has a doctor but no radiologist." Then "3 radiologists in Malawi, for 19.7 million people", with the source and its year. Then eight held-out test X-rays. | "A small clinic takes a chest X-ray, but no radiologist is there to read it. A 2022 survey counted three radiologists in Malawi. PneumoShift helps decide which X-ray a doctor sees first." |
| 0:14 to 0:45 | The demo | The screen recording of the demo (`docs/media/demo.mp4`): the page, the example "Adult (46), normal" is clicked, both model cards answer, then the heatmaps. A side panel repeats what the demo shows: 90.3% and "Not flagged, but not clearly normal", the warning, 5.4% and "No sign of pneumonia found". | "The live demo runs in the browser, so the X-ray never leaves the device. Two models answer side by side. We choose a test X-ray: an adult from another hospital, with a normal X-ray. The children-only model scores it 90 percent: high, but below its cut-off. The tool warns that this X-ray is unlike its training data. A second model, trained with adults too, says 5 percent. Each model also shows a heatmap of where it looked." |
| 0:45 to 0:57 | The demo | The demo screenshot `docs/media/example_adult_65.png`, both "Missed" marks ringed. Side panel: model A 85%, model B 11%, then cases caught on bedside X-rays (93%) and standing X-rays (28%). | "Both models miss this one: an adult with a lung opacity, on a standing X-ray. On standing X-rays, the children-only model catches only 28 percent of cases." |
| 0:57 to 1:19 | How it works | The model (ResNet-18, ImageNet, Hugging Face Hub, fine-tuned), the data (5,856 X-rays, children aged 1 to 5, one hospital; three test X-rays), then right answers on the 624 test X-rays: 93.8% against 62.5%. | "Both models are a ResNet-18 from the Hugging Face Hub, fine-tuned on chest X-rays. The usual tutorial dataset holds only children aged 1 to 5, from one hospital. On held-out children’s X-rays, the children-only model is right 94 times in 100. Always answering “pneumonia” would be right 63 times." |
| 1:19 to 1:40 | How it works | How we tested (split by patient, cut-offs fixed before testing, 7,694 X-rays of 3,307 patients). Bar chart of the ranking score: 0.975 on children; on adults 0.824, 0.874, baseline 0.848, 0.880; then the verdict with the difference and its interval. | "On adult X-rays from another hospital, the children-only model’s ranking score drops from 0.98 to 0.82. Adding 6,000 adult X-rays lifts it to 0.87. All 15,000 gave 0.88. By our rule, that is not a shown improvement." |
| 1:40 to 1:59 | Limits and next steps | Right answers on the adult test X-rays: 74.7% against 77.6%. Cards: the labels, also in the write-up (42%, heatmaps 25% and 52% against 57%), next step. Last: "This project is a prototype and is not medical advice." and the demo link. | "On adults, the children-only model is right 75 times in 100. Always answering “no” would be right 78 times. The adult labels are radiologists’ findings, not confirmed pneumonia. Next, we would test a third hospital. This is a research demo, not a medical device." |

## Before uploading

- The examples in the demo were drawn at random from the test X-rays, they were not hand-picked. You may say so.
- No names, no school, no faces in the video, its title or its description: round 1 is reviewed blind.
- Upload unlisted to YouTube, or to Google Drive with "anyone with the link can view". Open the link in a
  logged-out browser window, then add it to the first lines of the README.

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
