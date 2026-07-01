# PDFs

Place the 8–10 provided ESG/sustainability report PDFs in **this folder**.

- The **live UI upload flow** does not read this folder — it stores whatever you
  upload through the browser.
- The **offline experiment runner** (`python -m app.eval.experiments`) reads the
  PDFs from here, matching each file name against a key in
  `../ground_truth/ground_truth.json`.

So for the trend report to score correctly:

```
data/pdfs/acme_sustainability_2024.pdf      <-- file name ...
data/ground_truth/ground_truth.json         <-- ... matches a key here
```

This folder is git-ignored for blobs but the PDFs themselves should be
committed as part of the deliverable (the assignment asks for the provided PDFs
and your labels to be included).
