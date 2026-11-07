# Clearwell

Clearwell is a software tool that turns satellite flood data into a ranked list for district officials. It tells officials which villages most likely had their drinking water contaminated, so they can be tested and treated first.

To deliver this map to officals, Clearwell will be powered by two models. The first will create a floodmap across India, showing which areas are flooded in real-time. It will be powered by a U-Net convolutional neural network (CNN). That map will then be read by the second model, which will output a full risk score to officals so they can allocate their resources to areas that need them most. The second model will be powered by _.

## Repo Structure

xylem-clearwell/
├── README.md
├── requirements.txt
├── .gitignore        # data/, *.pt, __pycache__/, .env
├── config.py         # paths, region, dates, normalization numbers
├── train.py          # Phases 1-3
├── export_s1.py      # Phase 4
└── flood_map.py      # Phases 5-6