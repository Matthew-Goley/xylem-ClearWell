## Notes from the build machine (hand-written, contains no result numbers)

Delete `results/status_notes.md` after your first full run at home; it only describes the machine where the code was written.

- The code was written and checked in a cloud container with no GPU (CPU only, Python 3.12).
- Its network policy BLOCKED: planetarycomputer.microsoft.com (Sentinel-1/2, DEM, JRC on Planetary Computer), geodata.ucdavis.edu (GADM), overpass-api.de (OSM), WorldPop, the ImageNet encoder weights (download.pytorch.org, huggingface.co), data.gov.in and censusindia.gov.in.
- It could REACH: Google Cloud Storage (Sen1Floods11, Open Buildings, JRC GSW tiles), the AWS Copernicus DEM bucket, and geoBoundaries.
- Ran there for real: the full `get_sen1floods11` stage, and the full `train` + `test` stages on the CPU with a randomly initialised encoder (ImageNet weights blocked). Those Model 1 numbers are real but are a weaker CPU baseline; the GPU run at home replaces them.
- Also checked there: a smoke run of every stage that could run, the pytest suite on synthetic data (including an end-to-end synthetic test of flood_map -> villages -> risk -> validate -> summary), and real downloads through the DEM, JRC GSW, boundary and Open Buildings fallback code.
- NOT run there (untested on real data until the home run): `export_s1`, `flood_map`, `sanity_check`, `villages`, `risk` and `validate` on real Assam data, CUDA training with mixed precision, and validation (the IDSP file was not present).
