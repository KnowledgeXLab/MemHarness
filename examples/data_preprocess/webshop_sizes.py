# Copyright 2026 the MemAdaptor team.
"""Infer WebShop train/val placeholder counts without Gym or WebAgentTextEnv.

``WebshopMultiProcessEnv`` uses:

- Val: ``range(500)`` → 500 goals (indices 0–499).
- Train: ``range(500, len(goals))`` → ``len(goals) - 500``.

We replicate ``SimServer``'s ``load_products`` + ``get_goals`` *counts* using only
stdlib + JSON so ``prepare --infer_webshop_sizes`` does not import gym/torch/nltk/sklearn.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import warnings
from decimal import Decimal
from typing import Any


def _repo_root_from_here() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", ".."))


def _webshop_bundle_root() -> str:
    return os.path.join(
        _repo_root_from_here(),
        "agent_system",
        "environments",
        "env_package",
        "webshop",
        "webshop",
    )


def resolve_webshop_data_paths(use_small: bool) -> tuple[str, str]:
    """Same JSON paths as ``env_manager`` when building WebShop envs."""
    data_dir = os.path.join(_webshop_bundle_root(), "data")
    if use_small:
        fp = os.path.join(data_dir, "items_shuffle_1000.json")
        ap = os.path.join(data_dir, "items_ins_v2_1000.json")
    else:
        fp = os.path.join(data_dir, "items_shuffle.json")
        ap = os.path.join(data_dir, "items_ins_v2.json")
    return fp, ap


def _human_instructions_bundle_path() -> str:
    """Vendored copy under the repo (same as ``web_agent_site.utils.HUMAN_ATTR_PATH``)."""
    return os.path.join(_webshop_bundle_root(), "data", "items_human_ins.json")


def resolve_human_instructions_path(products_json_path: str) -> str:
    """Resolve ``items_human_ins.json``: prefer same directory as products JSON, then bundle."""
    sidecar = os.path.join(os.path.dirname(os.path.abspath(products_json_path)), "items_human_ins.json")
    if os.path.isfile(sidecar):
        return sidecar
    bundle = _human_instructions_bundle_path()
    if os.path.isfile(bundle):
        return bundle
    raise FileNotFoundError(
        "WebShop human instructions JSON (items_human_ins.json) not found. Put it next to your "
        f"items_shuffle*.json (expected {sidecar}), or restore the vendored file at {bundle}."
    )


def _clean_product_keys(products: list[dict[str, Any]]) -> None:
    for product in products:
        product.pop("product_information", None)
        product.pop("brand", None)
        product.pop("brand_url", None)
        product.pop("list_price", None)
        product.pop("availability_quantity", None)
        product.pop("availability_status", None)
        product.pop("total_reviews", None)
        product.pop("total_answered_questions", None)
        product.pop("seller_id", None)
        product.pop("seller_name", None)
        product.pop("fulfilled_by_amazon", None)
        product.pop("fast_track_message", None)
        product.pop("aplus_present", None)
        product.pop("small_description_old", None)


def _light_load_all_products(
    filepath: str,
    attrpath: str,
    *,
    human_ins_path: str,
    num_products: int | None,
    human_goals: bool,
) -> list[dict[str, Any]]:
    """Mirror ``web_agent_site.engine.engine.load_products`` outputs ``all_products`` only."""
    with open(filepath, encoding="utf-8") as f:
        products: list[dict[str, Any]] = json.load(f)
    _clean_product_keys(products)

    with open(human_ins_path, encoding="utf-8") as f:
        human_attributes = json.load(f)
    with open(attrpath, encoding="utf-8") as f:
        attributes = json.load(f)
    with open(human_ins_path, encoding="utf-8") as f:
        human_attributes = json.load(f)

    all_reviews: dict[str, Any] = {}
    all_ratings: dict[str, Any] = {}

    asins: set[str] = set()
    all_products: list[dict[str, Any]] = []

    if num_products is not None:
        products = products[:num_products]

    for i, p in enumerate(products):
        asin = p["asin"]
        if asin == "nan" or len(asin) > 10:
            continue
        if asin in asins:
            continue
        asins.add(asin)

        products[i]["category"] = p["category"]
        products[i]["query"] = p["query"]
        products[i]["product_category"] = p["product_category"]

        products[i]["Title"] = p["name"]
        products[i]["Description"] = p["full_description"]
        products[i]["Reviews"] = all_reviews.get(asin, [])
        products[i]["Rating"] = all_ratings.get(asin, "N.A.")
        for r in products[i]["Reviews"]:
            if "score" not in r:
                r["score"] = r.pop("stars")
            if "review" not in r:
                r["body"] = ""
            else:
                r["body"] = r.pop("review")
        products[i]["BulletPoints"] = (
            p["small_description"]
            if isinstance(p["small_description"], list)
            else [p["small_description"]]
        )

        pricing = p.get("pricing")
        if pricing is None or not pricing:
            pricing = [100.0]
            price_tag = "$100.0"
        else:
            pricing = [
                float(Decimal(re.sub(r"[^\d.]", "", price))) for price in pricing.split("$")[1:]
            ]
            if len(pricing) == 1:
                price_tag = f"${pricing[0]}"
            else:
                price_tag = f"${pricing[0]} to ${pricing[1]}"
                pricing = pricing[:2]
        products[i]["pricing"] = pricing
        products[i]["Price"] = price_tag

        options: dict[str, list[str]] = {}
        option_to_image: dict[str, Any] = {}
        customization_options = p["customization_options"]
        if customization_options:
            for option_name, option_contents in customization_options.items():
                if option_contents is None:
                    continue
                option_name = option_name.lower()
                option_values: list[str] = []
                for option_content in option_contents:
                    option_value = option_content["value"].strip().replace("/", " | ").lower()
                    option_image = option_content.get("image", None)
                    option_values.append(option_value)
                    option_to_image[option_value] = option_image
                options[option_name] = option_values
        products[i]["options"] = options
        products[i]["option_to_image"] = option_to_image

        if asin in attributes and "attributes" in attributes[asin]:
            products[i]["Attributes"] = attributes[asin]["attributes"]
        else:
            products[i]["Attributes"] = ["DUMMY_ATTR"]

        if human_goals:
            if asin in human_attributes:
                products[i]["instructions"] = human_attributes[asin]
        else:
            products[i]["instruction_text"] = attributes[asin].get("instruction", None)
            products[i]["instruction_attributes"] = attributes[asin].get("instruction_attributes", None)

        products[i]["MainImage"] = p["images"][0]
        products[i]["query"] = p["query"].lower().strip()

        all_products.append(products[i])

    return all_products


def _count_goals_human(all_products: list[dict[str, Any]]) -> int:
    """Match ``get_human_goals`` goal list length (excluding empty-attribute instructions)."""
    n = 0
    for item in all_products:
        if "instructions" not in item:
            continue
        for instr in item["instructions"]:
            attrs = instr["instruction_attributes"]
            if len(attrs) == 0:
                continue
            n += 1
    return n


def _count_goals_synthetic(all_products: list[dict[str, Any]]) -> int:
    """Match ``get_synthetic_goals`` goal list length."""
    n = 0
    for product in all_products:
        if "instruction_text" not in product or product["instruction_text"] is None:
            continue
        attrs = product["instruction_attributes"]
        if len(attrs) == 0:
            continue
        options = product["options"]
        option_names = sorted(options)
        combinations = list(itertools.product(*(options[oname] for oname in option_names)))
        n += len(combinations)
    return n


def infer_webshop_num_tasks(
    *,
    use_small: bool = False,
    human_goals: bool = False,
    observation_mode: str = "text",
    num_products: int | None = None,
    file_path: str | None = None,
    attr_path: str | None = None,
) -> tuple[int, int]:
    """
    Return ``(num_train_goals, num_val_goals)`` consistent with ``WebshopMultiProcessEnv``.

    ``observation_mode`` is accepted for API compatibility with the old Gym-based helper; it is unused.
    """
    del observation_mode  # unused; kept for call-site compatibility

    if bool(file_path) ^ bool(attr_path):
        raise ValueError("webshop file_path and attr_path must both be set or both omitted")

    if file_path is None:
        file_path, attr_path = resolve_webshop_data_paths(use_small)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"WebShop products JSON not found: {file_path}")
    if not os.path.isfile(attr_path):
        raise FileNotFoundError(f"WebShop attributes JSON not found: {attr_path}")

    human_ins = resolve_human_instructions_path(file_path)

    all_products = _light_load_all_products(
        file_path,
        attr_path,
        human_ins_path=human_ins,
        num_products=num_products,
        human_goals=human_goals,
    )
    n_goals = _count_goals_human(all_products) if human_goals else _count_goals_synthetic(all_products)

    if n_goals < 500:
        raise ValueError(
            f"Inferred WebShop goal count is {n_goals} (<500). WebshopMultiProcessEnv assumes val uses "
            "indices 0–499 and train uses the rest; fewer goals makes rollout indexing invalid. "
            "Common causes: (1) --webshop-human-goals does not match env.webshop.human_goals "
            "(ppo_trainer default is synthetic → use --no-webshop-human-goals / omit --webshop-human-goals); "
            "(2) corrupt or truncated items_shuffle / items_ins_v2 / items_human_ins."
        )

    n_val = len(range(500))
    n_train = len(range(500, n_goals))
    return n_train, n_val
