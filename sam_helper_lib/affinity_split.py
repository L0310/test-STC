import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .image_ops import (
    _connected_components,
    _ensure_binary_mask,
    _ensure_uint8_rgb,
    _filter_components,
    _label_map_to_components,
    _normalize_gray_map,
)


class SAMTrainAffinityMixin:
    def _load_depth_map(self, sample_name: str, out_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if not self.depth_index:
            return None
        stem = os.path.splitext(os.path.basename(sample_name))[0]
        depth_path = self.depth_index.get(stem)
        if depth_path is None:
            return None
        depth_gray = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_gray is None:
            return None
        depth_map = _normalize_gray_map(depth_gray)
        if depth_map.shape != tuple(out_shape):
            depth_map = cv2.resize(depth_map, out_shape[::-1], interpolation=cv2.INTER_LINEAR)
            depth_map = _normalize_gray_map(depth_map)
        return depth_map.astype(np.float32)

    def _get_dino_extractor(self):
        if self._dino_checked:
            return self._dino_extractor
        self._dino_checked = True
        if not self.dino_weight:
            return None
        if not os.path.exists(self.dino_weight):
            print("Warn: DINO semantic affinity checkpoint not found: {}. Using depth/RGB affinity only.".format(self.dino_weight))
            return None
        try:
            from depth_affinity_spectral_demo import _OnTheFlyDINOExtractor, _default_dino_repo

            self._dino_extractor = _OnTheFlyDINOExtractor(
                weight_path=Path(self.dino_weight),
                model_name=self.dino_model,
                repo_path=self.dino_repo or _default_dino_repo(),
                device=self.dino_device,
                max_side=self.dino_max_side,
            )
            print("Using DINO semantic affinity from {}.".format(os.path.abspath(self.dino_weight)))
        except Exception as exc:
            self._dino_extractor = None
            print("Warn: failed to initialize DINO semantic affinity extractor: {}. Using depth/RGB affinity only.".format(exc))
        return self._dino_extractor

    def _extract_dino_appearance_map(
        self,
        image_rgb: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        extractor = self._get_dino_extractor()
        if extractor is None:
            return None
        try:
            return extractor.extract(image_rgb, target_shape=target_shape)
        except Exception as exc:
            print("Warn: failed to extract DINO semantic affinity features: {}. Using depth/RGB affinity only for this sample.".format(exc))
            return None

    def _split_prompt_mask_with_local_affinity(
        self,
        prompt_mask: np.ndarray,
        image_rgb: np.ndarray,
        depth_map: Optional[np.ndarray],
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        prompt_mask = _ensure_binary_mask(prompt_mask)
        if int(prompt_mask.sum()) <= 0:
            return [], None
        try:
            from skimage.segmentation import slic
        except Exception:
            return _connected_components(prompt_mask), None

        rgb = _ensure_uint8_rgb(image_rgb)
        if rgb.shape[:2] != prompt_mask.shape:
            rgb = cv2.resize(rgb, prompt_mask.shape[::-1], interpolation=cv2.INTER_LINEAR)
        try:
            full_image_labels = slic(
                rgb,
                n_segments=self.affinity_superpixel_count,
                compactness=self.affinity_slic_compactness,
                sigma=self.affinity_slic_sigma,
                start_label=1,
                enforce_connectivity=True,
                min_size_factor=0.4,
                max_size_factor=3.0,
                channel_axis=-1,
            ).astype(np.int32)
        except Exception:
            return _connected_components(prompt_mask), None

        label_map = np.zeros_like(prompt_mask, dtype=np.int32)
        next_label = 1
        for component in _connected_components(prompt_mask):
            component = _ensure_binary_mask(component)
            area = int(component.sum())
            if area < self.affinity_min_component_area:
                label_map[component > 0] = next_label
                next_label += 1
                continue
            ys, xs = np.where(component > 0)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            crop_slice = np.s_[y0:y1 + 1, x0:x1 + 1]
            component_crop = component[crop_slice]
            labels_crop = full_image_labels[crop_slice]
            if int(labels_crop.max()) <= 1:
                label_map[component > 0] = next_label
                next_label += 1
                continue

            accepted = 0
            for local_label in sorted(int(value) for value in np.unique(labels_crop) if int(value) > 0):
                masked_region = ((labels_crop == local_label) & (component_crop > 0)).astype(np.uint8)
                for region in _connected_components(masked_region):
                    region = _filter_components(region, min_area=self.affinity_min_superpixel_area)
                    if int(region.sum()) < self.affinity_min_instance_area:
                        continue
                    label_view = label_map[crop_slice]
                    label_view[region > 0] = next_label
                    next_label += 1
                    accepted += 1
            if accepted == 0:
                label_map[component > 0] = next_label
                next_label += 1

        return _label_map_to_components(label_map, min_area=max(16, int(self.affinity_min_instance_area // 2))), label_map

    def _split_prompt_mask_into_affinity_instances(
        self,
        prompt_mask: np.ndarray,
        image_rgb: np.ndarray,
        depth_map: Optional[np.ndarray],
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        prompt_mask = _ensure_binary_mask(prompt_mask)
        if int(prompt_mask.sum()) <= 0:
            return [], None
        depth = (
            _normalize_gray_map(depth_map)
            if depth_map is not None
            else np.zeros(prompt_mask.shape, dtype=np.float32)
        )
        appearance_map = self._extract_dino_appearance_map(image_rgb, target_shape=prompt_mask.shape)
        try:
            from depth_affinity_spectral_demo import split_depth_instances_affinity_spectral

            results = split_depth_instances_affinity_spectral(
                depth=depth,
                gt_mask=prompt_mask,
                rgb=image_rgb,
                appearance_map=appearance_map,
                min_component_area=self.affinity_min_component_area,
                min_instance_area=self.affinity_min_instance_area,
                median_ksize=5,
                bilateral_d=7,
                bilateral_sigma_color=25.0,
                bilateral_sigma_space=25.0,
                superpixel_count=self.affinity_superpixel_count,
                min_superpixel_area=self.affinity_min_superpixel_area,
                slic_compactness=self.affinity_slic_compactness,
                slic_sigma=self.affinity_slic_sigma,
                slic_depth_scale=self.affinity_slic_depth_scale,
                slic_input_mode="rgb",
                dino_pca_dim=self.dino_pca_dim,
                sigma_sem=self.affinity_sigma_sem,
                sigma_dep=self.affinity_sigma_dep,
                sigma_spatial=self.affinity_sigma_spatial,
                sigma_edge=self.affinity_sigma_edge,
                min_affinity=self.affinity_min_affinity,
                min_cluster_regions=self.affinity_min_cluster_regions,
                ncut_threshold=self.affinity_ncut_threshold,
                max_recursion_depth=self.affinity_max_recursion_depth,
            )
            components = _label_map_to_components(
                results.get("label_map", np.zeros_like(prompt_mask, dtype=np.int32)),
                min_area=max(16, int(self.affinity_min_instance_area // 2)),
            )
            if components:
                superpixel_label_map = np.asarray(
                    results.get("superpixel_label_map", np.zeros_like(prompt_mask, dtype=np.int32)),
                    dtype=np.int32,
                )
                return components, superpixel_label_map
        except Exception as exc:
            if not hasattr(self, "_affinity_import_warned"):
                self._affinity_import_warned = True
                print(
                    "Warn: depth_affinity_spectral_demo is unavailable ({}). "
                    "Using local RGB/depth SLIC split fallback.".format(exc)
                )

        components, superpixel_label_map = self._split_prompt_mask_with_local_affinity(prompt_mask, image_rgb, depth_map)
        return (components if components else _connected_components(prompt_mask)), superpixel_label_map
