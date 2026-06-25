import pandas as pd
import numpy as np
from pathlib import Path
import random

class Baselines():

    random.seed(1588)

    TRAIN_PATH = Path("train_data/train_labels_t_ns.csv")

    ALL_CROP_COLS = [
        "barley_production_t",
        "canola_production_t",
        "corn_bu_production_t",
        "corn_tons_production_t",
        "cotton_production_t",
        "hay_production_t",
        "oats_production_t",
        "peanuts_production_t",
        "rice_production_t",
        "sorghum_production_t",
        "soybeans_production_t",
        "wheat_production_t",
    ]

    CROP_NAMES = [
    "barley", "canola", "corn_bu", "corn_tons", "cotton",
    "hay", "oats", "peanuts", "rice", "sorghum", "soybeans", "wheat",
    ]

    TILE_AREA = 0.48 * 0.48


    def __init__(self):
        super().__init__()

        rename_map = dict(zip(self.ALL_CROP_COLS, self.CROP_NAMES))
        train_set = pd.read_csv(self.TRAIN_PATH)
        self.train_set = train_set.rename(columns=rename_map)
        self.crop_cols = [c for c in self.CROP_NAMES if c in self.train_set.columns]

    def dominant_class_normalised_average_baseline(self):
        class_map = self.train_set[self.crop_cols].notna()
        crop_counts = class_map.sum()
        max_class = crop_counts.idxmax()
        # Normalising the values by the county size to get the per tile average
        normalised_production = self.train_set[max_class] / self.train_set["area_km2"]
        tile_normalised_production = normalised_production / self.TILE_AREA
        normalised_average = tile_normalised_production.mean(skipna=True)
        return max_class, normalised_average

    def dominant_class_field_adjusted_normalised_average_baseline(self, field_size_ha):
        class_map = self.train_set[self.crop_cols].notna()
        crop_counts = class_map.sum()
        max_class = crop_counts.idxmax()
        # Normalising the values by the county size to get the per tile average
        normalised_production = self.train_set[max_class] / self.train_set["area_km2"]
        tile_normalised_production = normalised_production / self.TILE_AREA
        normalised_average = tile_normalised_production.mean(skipna=True)
        field_adjusted_normalised_average = normalised_average / field_size_ha
        return max_class, field_adjusted_normalised_average

    def right_class_normalised_class_average(self, crop_class):
        normalised_production_per_class = self.train_set[self.crop_cols].div(
            self.train_set["area_km2"], axis=0
        )
        tile_normalised_production_per_class = normalised_production_per_class[self.crop_cols].div(self.TILE_AREA)
        per_crop_means = tile_normalised_production_per_class.mean(skipna=True)
        return crop_class, per_crop_means[crop_class] 
    
    def right_class_field_adjusted_normalised_class_average(self, crop_class, field_size_ha):
        normalised_production_per_class = self.train_set[self.crop_cols].div(
            self.train_set["area_km2"], axis=0
        )
        tile_normalised_production_per_class = normalised_production_per_class[self.crop_cols].div(self.TILE_AREA)
        per_crop_means = tile_normalised_production_per_class.mean(skipna=True)
        filed_adjusted = per_crop_means[crop_class] / field_size_ha
        return crop_class, filed_adjusted 

    def random_class_normalised_dataset_average(self):    
        normalised_production_per_class = self.train_set[self.crop_cols].div(
            self.train_set["area_km2"], axis=0
        )
        random_class = random.choice(self.crop_cols)
        tile_normalised_production_per_class = normalised_production_per_class[self.crop_cols].div(self.TILE_AREA)
        dataset_average = tile_normalised_production_per_class.mean(skipna=True).mean(skipna=True)
        return random_class, dataset_average

    def random_class_normalised_class_average(self):
        normalised_production_per_class = self.train_set[self.crop_cols].div(
            self.train_set["area_km2"], axis=0
        )
        random_class = random.choice(self.crop_cols)
        tile_normalised_production_per_class = normalised_production_per_class[self.crop_cols].div(self.TILE_AREA)
        per_crop_means = tile_normalised_production_per_class.mean(skipna=True)
        return random_class, per_crop_means[random_class]
    
    def random_class_field_adjusted_normalised_class_average(self, field_size):
        normalised_production_per_class = self.train_set[self.crop_cols].div(
            self.train_set["area_km2"], axis=0
        )
        random_class = random.choice(self.crop_cols)
        tile_normalised_production_per_class = normalised_production_per_class[self.crop_cols].div(self.TILE_AREA)
        per_crop_means = tile_normalised_production_per_class.mean(skipna=True)
        field_size_adjusted_production = per_crop_means[random_class] / field_size
        return random_class, field_size_adjusted_production
        

    # Baselines:
        # Just the dominent class and the normalised average
        # Just the dominant class and the field adjusted normalised average
        # Always the right class and the normalised average
        # Always the right class and the field adjusted normalised average
        # Random class and the dataset normalised average
        # Always the right class and the dataset normalised average