"""
config.py - every setting for the Clearwell pipeline lives in this one file.

Change values here, not inside the scripts. Nothing in this file is a result:
numbers computed from data (for example the radar normalization numbers) are
written by code to results/norm_stats.json and loaded by load_norm_stats().

Smoke mode: every script accepts --smoke. Smoke mode writes to outputs/smoke/
and results/smoke/ so a tiny test run never mixes with the real results.
"""

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent


@dataclass
class Paths:
    """All folders and important files for one run (full or smoke)."""

    root: Path
    smoke: bool

    @property
    def data(self):
        return self.root / "data"

    @property
    def manual(self):
        # Files you add by hand (your friend's IDSP csv, census table).
        return self.data / "manual"

    @property
    def sen1floods11(self):
        # Shared between smoke and full runs (smoke just uses fewer tiles).
        return self.data / "sen1floods11"

    @property
    def boundaries(self):
        return self.data / "boundaries"

    @property
    def assam(self):
        # Smoke mode downloads a tiny area, so keep it separate.
        return self.data / ("assam_smoke" if self.smoke else "assam")

    @property
    def cache(self):
        # Downloads used by Model 2 (OSM, WorldPop, buildings, DEM tiles).
        return self.data / "cache"

    @property
    def outputs(self):
        base = self.root / "outputs"
        return base / "smoke" if self.smoke else base

    @property
    def results(self):
        base = self.root / "results"
        return base / "smoke" if self.smoke else base

    @property
    def logs(self):
        return self.outputs / "logs"

    @property
    def markers(self):
        return self.outputs / "markers"

    @property
    def norm_stats(self):
        return self.results / "norm_stats.json"

    @property
    def checkpoint(self):
        return self.outputs / "unet.pt"

    @property
    def aoi(self):
        # Written by export_s1: chosen districts, orbit, dates, pixel grid.
        return self.outputs / "aoi.json"

    def make_dirs(self):
        for d in [self.data, self.manual, self.sen1floods11, self.boundaries,
                  self.assam, self.cache, self.outputs, self.results,
                  self.logs, self.markers]:
            d.mkdir(parents=True, exist_ok=True)


def get_paths(smoke=False, root=None):
    """Return a Paths object. Tests pass root=tmp_path to stay out of the repo."""
    paths = Paths(root=Path(root) if root else ROOT, smoke=bool(smoke))
    paths.make_dirs()
    return paths


# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
SEED = 42
IS_WINDOWS = platform.system() == "Windows"
# DataLoader workers: Windows multiprocessing often hangs with workers > 0.
NUM_WORKERS = 0 if IS_WINDOWS else 4

# ---------------------------------------------------------------------------
# MODEL 1, phase 1: Sen1Floods11 data
# ---------------------------------------------------------------------------
S1F11_BUCKET = "sen1floods11"
S1F11_HAND_PREFIX = "v1.1/data/flood_events/HandLabeled/"
S1F11_FOLDERS = ["S1Hand", "LabelHand", "S1OtsuLabelHand"]
S1F11_SPLITS_PREFIX = "v1.1/splits/flood_handlabeled/"
S1F11_SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}
# Smoke mode: how many tiles to use per split.
SMOKE_TILES = {"train": 8, "valid": 3, "test": 4, "bolivia": 2}

# Radar preprocessing. dB values are clipped to this range before
# normalization. Sen1Floods11 radar and the Assam imagery both go through this.
CLIP_DB_MIN = -50.0
CLIP_DB_MAX = 5.0

# ---------------------------------------------------------------------------
# MODEL 1, phase 2: training
# ---------------------------------------------------------------------------
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"   # falls back to None (random init) if download fails
BATCH_SIZE = 8
BATCH_SIZE_OOM_FALLBACK = 4    # used automatically after a CUDA out-of-memory error
CROP_SIZE = 256                # random training crops; validation uses full 512 tiles
LEARNING_RATE = 3e-4           # AdamW, one rate for all layers, cosine schedule
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 10       # epochs without val IoU improvement
USE_AMP = True                 # mixed precision, only used on CUDA

# ---------------------------------------------------------------------------
# MODEL 1, phase 4: Assam Sentinel-1 export (Microsoft Planetary Computer)
# ---------------------------------------------------------------------------
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S1_COLLECTION = "sentinel-1-rtc"           # first choice (terrain corrected gamma0)
S1_FALLBACK_COLLECTION = "sentinel-1-grd"  # only used if RTC search fails

STATE_NAME = "Assam"
FLOOD_YEAR = 2020
FLOOD_START = "2020-06-25"
FLOOD_END = "2020-08-10"
# Normal (non-flood) years: same calendar dates as the flood window.
NORMAL_YEARS = [2018, 2021]
EXTRA_NORMAL_YEARS = [2019]  # added only with --extra-normal

# Candidate districts (Brahmaputra flood-prone). "auto" picks 2 or 3 adjacent
# ones with the best single-orbit Sentinel-1 coverage. You can also set an
# explicit list, for example DISTRICTS = ["Dhemaji", "Lakhimpur", "Majuli"].
CANDIDATE_DISTRICTS = ["Dhemaji", "Lakhimpur", "Majuli", "Jorhat",
                       "Barpeta", "Nalbari", "Morigaon"]
DISTRICTS = "auto"
# Used when "auto" cannot run (for example no boundaries available).
DEFAULT_DISTRICTS = ["Dhemaji", "Lakhimpur", "Majuli"]
MAX_DISTRICTS = 3

# Known spelling variants (lower case key -> name used by boundary files).
# Used for boundary files and for matching names in the IDSP file.
DISTRICT_ALIASES = {
    "marigaon": "Morigaon",
    "morigaon": "Morigaon",
    "north lakhimpur": "Lakhimpur",
    "sibsagar": "Sivasagar",
    "sivsagar": "Sivasagar",
    "kamrup metro": "Kamrup Metropolitan",
    "kamrup (m)": "Kamrup Metropolitan",
    "kamrup metropolitan": "Kamrup Metropolitan",
    "kamrup (r)": "Kamrup",
    "kamrup rural": "Kamrup",
    "north cachar hills": "Dima Hasao",
    "n c hills": "Dima Hasao",
    "karbi anglong east": "Karbi Anglong",
}

# Boundary sources, tried in order.
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_2.json.zip"
GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/IND/{level}/"
GEOBOUNDARIES_DIRECT = ("https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/"
                        "main/releaseData/gbOpen/IND/{level}/geoBoundaries-IND-{level}_simplified.geojson")
# Last resort if no boundary file can be downloaded: an approximate bounding
# box (lon_min, lat_min, lon_max, lat_max) around Dhemaji, Lakhimpur and
# Majuli. This is only an area to download, not an official boundary.
AOI_BBOX_FALLBACK = (93.6, 26.7, 95.6, 27.9)

# Smoke mode: a tiny box (about 5 km x 5 km) centred in Majuli, chosen so that
# roughly a third of it is permanent river water in JRC Global Surface Water.
SMOKE_BBOX = (94.325, 26.95, 94.375, 26.995)
SMOKE_FLOOD_DATES = 2
SMOKE_NORMAL_DATES = 2

PIXEL_SIZE_M = 10          # output grid resolution in metres
AOI_BUFFER_M = 500         # padding added around the district boundaries
MIN_AOI_COVERAGE = 0.95    # a date counts only if its scenes cover this share of the AOI
EXPORT_CHUNK = 2048        # pixels per side when writing GeoTIFFs in pieces
NETWORK_RETRIES = 4
WATERISH_VV_DB = -18.0     # crude "looks like water" threshold, only for a check
NORMAL_FLOOD_RATIO_WARN = 0.8  # warn if a normal year has >= 80% of the flood-window water

# ---------------------------------------------------------------------------
# MODEL 1, phase 5: flood mapping
# ---------------------------------------------------------------------------
INFER_TILE = 512
INFER_OVERLAP = 64
INFER_BATCH = 8
INFER_BLOCK = 4096            # big raster is processed in blocks of this size
WATER_THRESHOLD = 0.5
TTA_HFLIP = False             # horizontal-flip test time augmentation
NORMAL_WATER_MIN_FRAC = 0.5   # "normal water" if water in >= half of normal passes
MIN_FLOOD_BLOB_HA = 0.5       # remove flood blobs smaller than this (hectares)
PLAUSIBILITY_DOWNSAMPLE = 5   # 10 m -> 50 m grid for the distance check
PLAUSIBILITY_PERMUTATIONS = 200

# ---------------------------------------------------------------------------
# MODEL 1, phase 6: sanity checks
# ---------------------------------------------------------------------------
S2_COLLECTION = "sentinel-2-l2a"
S2_MAX_DAYS = 3
S2_MAX_CLOUD = 20
S2_MIN_CLEAR_FRACTION = 0.2   # need at least 20% of the AOI cloud free
JRC_COLLECTION = "jrc-gsw"
JRC_OCCURRENCE_MIN = 50       # GSW occurrence (percent) counted as permanent water
JRC_GCS_URL = ("https://storage.googleapis.com/global-surface-water/downloads2021/"
               "occurrence/occurrence_{lon}_{lat}v1_4_2021.tif")

# ---------------------------------------------------------------------------
# MODEL 2: village proxies and features
# ---------------------------------------------------------------------------
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
MIN_OSM_POINTS = 100          # fewer than this -> fall back to a 1 km grid
VILLAGE_MAX_RADIUS_M = 1500   # Voronoi cells are clipped to this radius
GRID_CELL_M = 1000            # fallback grid cell size
FIRST_FLOOD_MIN_FRAC = 0.01   # unit counts as "flooded on date D" if >= 1% flooded

# WorldPop 2020 population (100 m). The REST API is asked for the file URL;
# the direct URLs are fallbacks. A file you download yourself to
# data/manual/worldpop.tif is used first.
WORLDPOP_API = "https://hub.worldpop.org/rest/data/pop/{alias}?iso3=IND"
WORLDPOP_ALIASES = ["cic2020_100m", "cic2020_UNadj_100m", "wpgp"]
WORLDPOP_DIRECT_URLS = [
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/IND/ind_ppp_2020_constrained.tif",
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/ind_ppp_2020.tif",
]

# Google Open Buildings v3. "points" files hold one row per building
# (centroid, area, confidence), which is enough to count buildings.
OPEN_BUILDINGS_URL = ("https://storage.googleapis.com/open-buildings-data/v3/"
                      "{kind}_s2_level_4_gzip/{token}_buildings.csv.gz")
OPEN_BUILDINGS_KIND = "points"
OPEN_BUILDINGS_MIN_CONFIDENCE = 0.7
OPEN_BUILDINGS_MAX_GB = 6.0   # skip buildings if the files are bigger than this

# Copernicus DEM 30 m: Planetary Computer first, then the public AWS bucket.
DEM_COLLECTION = "cop-dem-glo-30"
DEM_AWS_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/"
               "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
               "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")

# Census 2011 drinking water source. Fill this file by hand (see README) with
# columns: district, pct_shallow_well_handpump. If it is missing the
# feature is left out of the risk index.
CENSUS_WATER_FILE = "census_water.csv"
CENSUS_WATER_TEMPLATE = "census_water_template.csv"

# ---------------------------------------------------------------------------
# MODEL 2: risk index. Weights fixed in advance, NOT tuned on outcomes.
# ---------------------------------------------------------------------------
RISK_WEIGHTS = {
    "exposure": 0.35,        # flooded_frac
    "duration": 0.25,        # max_duration_days
    "exposed_people": 0.15,  # population (or buildings) in flooded pixels
    "well_reliance": 0.15,   # district share of shallow wells / hand pumps
    "low_elevation": 0.10,   # 1 - elev_pctile
}
TOP_K_MAP = 20
TOP_K_EVOLUTION = 50

# ---------------------------------------------------------------------------
# MODEL 2: validation
# ---------------------------------------------------------------------------
IDSP_FILE = "idsp_weather.csv"
OUTCOME_WEEKS_AFTER_FLOOD = 4
FUZZY_MATCH_MIN_SCORE = 85
FILTER_WATERBORNE = True
WATERBORNE_KEYWORDS = ["diarr", "cholera", "typhoid", "enteric", "dysentery",
                       "gastro", "hepatitis a", "hepatitis-a", "hepatitis e",
                       "hepatitis-e", "jaundice", "add"]
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 10000
RUN_LEARNED_MODEL = False
LEARNED_MIN_ROWS = 300
LEARNED_MIN_POSITIVES = 20


# ---------------------------------------------------------------------------
# Loading generated numbers
# ---------------------------------------------------------------------------
def load_norm_stats(paths):
    """Load the VV/VH mean and std computed from Sen1Floods11 TRAIN tiles.

    The file is written by train.py (stage get_sen1floods11). The numbers are
    never typed by hand.
    """
    path = paths.norm_stats
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist yet. Run: python run_all.py --stage get_sen1floods11"
            + (" --smoke" if paths.smoke else ""))
    with open(path) as f:
        stats = json.load(f)
    return stats


def environment_summary():
    """Small dict describing this machine (used in logs and SUMMARY.md)."""
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            info["gpu_free_vram_gb"] = round(free / 1e9, 2)
            info["gpu_total_vram_gb"] = round(total / 1e9, 2)
            info["cuda_version"] = torch.version.cuda
    except Exception as err:  # torch missing or broken
        info["torch_error"] = str(err)
    return info
