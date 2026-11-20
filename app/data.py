from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Product:
    product_id: str
    name: str
    category: str


@dataclass(frozen=True)
class Session:
    session_id: str
    items: tuple[str, ...]
    pattern: str


CATALOGUE = (
    Product("dairy_milk", "Milk", "dairy"), Product("dairy_butter", "Butter", "dairy"),
    Product("dairy_cheddar", "Cheddar Cheese", "dairy"), Product("dairy_yogurt", "Yogurt", "dairy"),
    Product("dairy_eggs", "Eggs", "dairy"), Product("dairy_cream", "Cream", "dairy"),
    Product("dairy_mozzarella", "Mozzarella", "dairy"), Product("dairy_cottage", "Cottage Cheese", "dairy"),
    Product("bakery_bread", "Bread", "bakery"), Product("bakery_buns", "Burger Buns", "bakery"),
    Product("bakery_bagel", "Bagels", "bakery"), Product("bakery_croissant", "Croissants", "bakery"),
    Product("bakery_pita", "Pita Bread", "bakery"), Product("bakery_tortilla", "Tortillas", "bakery"),
    Product("bakery_baguette", "Baguette", "bakery"), Product("bakery_rolls", "Dinner Rolls", "bakery"),
    Product("bev_coffee", "Coffee", "beverages"), Product("bev_tea", "Tea", "beverages"),
    Product("bev_orange", "Orange Juice", "beverages"), Product("bev_soda", "Soft Drinks", "beverages"),
    Product("bev_water", "Sparkling Water", "beverages"), Product("bev_juice", "Apple Juice", "beverages"),
    Product("fruit_banana", "Bananas", "fruit"), Product("fruit_apple", "Apples", "fruit"),
    Product("fruit_berries", "Mixed Berries", "fruit"), Product("fruit_lemon", "Lemons", "fruit"),
    Product("fruit_grapes", "Grapes", "fruit"), Product("fruit_avocado", "Avocados", "fruit"),
    Product("veg_tomato", "Tomatoes", "vegetables"), Product("veg_onion", "Onions", "vegetables"),
    Product("veg_spinach", "Spinach", "vegetables"), Product("veg_pepper", "Bell Peppers", "vegetables"),
    Product("veg_carrot", "Carrots", "vegetables"), Product("veg_potato", "Potatoes", "vegetables"),
    Product("snack_chips", "Chips", "snacks"), Product("snack_crackers", "Crackers", "snacks"),
    Product("snack_chocolate", "Chocolate", "snacks"), Product("snack_nuts", "Mixed Nuts", "snacks"),
    Product("snack_cereal", "Cereal", "snacks"), Product("snack_jam", "Jam", "snacks"),
    Product("pantry_sugar", "Sugar", "pantry"),
    Product("meat_patties", "Burger Patties", "meat_alternatives"), Product("meat_chicken", "Chicken Breast", "meat_alternatives"),
    Product("meat_tofu", "Tofu", "meat_alternatives"), Product("meat_lentils", "Lentils", "meat_alternatives"),
    Product("meat_turkey", "Turkey Slices", "meat_alternatives"), Product("meat_sausage", "Sausages", "meat_alternatives"),
    Product("pantry_pasta", "Pasta", "pantry"), Product("pantry_sauce", "Pasta Sauce", "pantry"),
    Product("pantry_rice", "Rice", "pantry"), Product("pantry_spices", "Spices", "pantry"),
    Product("home_toothpaste", "Toothpaste", "personal_care"), Product("home_toothbrush", "Toothbrush", "personal_care"),
    Product("home_detergent", "Laundry Detergent", "household"), Product("home_towels", "Paper Towels", "household"),
    Product("frozen_peas", "Frozen Peas", "frozen"), Product("frozen_pizza", "Frozen Pizza", "frozen"),
)

PATTERNS: tuple[tuple[str, tuple[str, ...], float], ...] = (
    ("toast_breakfast", ("bakery_bread", "dairy_butter", "snack_jam"), 0.13),
    ("pasta_night", ("pantry_pasta", "pantry_sauce", "dairy_mozzarella"), 0.13),
    ("coffee_morning", ("bev_coffee", "dairy_milk", "pantry_sugar"), 0.07),
    ("movie_night", ("snack_chips", "bev_soda", "snack_chocolate"), 0.11),
    ("lentil_dinner", ("pantry_rice", "meat_lentils", "pantry_spices"), 0.11),
    ("cereal_breakfast", ("snack_cereal", "dairy_milk", "fruit_banana"), 0.11),
    ("burger_meal", ("bakery_buns", "meat_patties", "dairy_cheddar"), 0.11),
    ("oral_care", ("home_toothpaste", "home_toothbrush"), 0.08),
    ("salad_lunch", ("veg_spinach", "fruit_avocado", "veg_tomato"), 0.07),
    ("snack_board", ("bakery_baguette", "dairy_cheddar", "fruit_grapes"), 0.06),
)
PATTERNS = tuple(row for row in PATTERNS if row[2] > 0)
INDEPENDENT_RATE = 0.27
UNUSUAL_RATE = 0.08


class DataError(ValueError):
    pass


def catalogue_rows() -> list[dict[str, str]]:
    return [asdict(product) for product in CATALOGUE]


def product_lookup() -> dict[str, Product]:
    return {product.product_id: product for product in CATALOGUE}


def generate_sessions(count: int = 2400, seed: int = 1729) -> list[Session]:
    if count < 10:
        raise DataError("count must be at least 10 to create useful train, validation, and test splits")
    rng = random.Random(seed)
    items = [product.product_id for product in CATALOGUE]
    pattern_names, pattern_items, pattern_weights = zip(*PATTERNS)
    sessions: list[Session] = []
    for index in range(count):
        draw = rng.random()
        if draw < INDEPENDENT_RATE:
            size = rng.randint(2, 8)
            chosen = rng.sample(items, size)
            label = "independent"
        elif draw < INDEPENDENT_RATE + UNUSUAL_RATE:
            size = rng.randint(2, 5)
            chosen = rng.sample(items, size)
            label = "unusual"
        else:
            pattern_index = rng.choices(range(len(PATTERNS)), weights=pattern_weights, k=1)[0]
            core = list(pattern_items[pattern_index])
            if rng.random() < 0.14 and len(core) > 2:
                core.pop(rng.randrange(len(core)))
            extras_count = rng.randint(0, max(0, 7 - len(core)))
            remaining = [item for item in items if item not in core]
            extras = rng.sample(remaining, min(extras_count, len(remaining)))
            chosen = core + extras
            label = pattern_names[pattern_index]
        if not 2 <= len(chosen) <= 8:
            raise DataError("generated basket length fell outside the configured range")
        sessions.append(Session(f"session_{index:06d}", tuple(chosen), label))
    return sessions


def split_sessions(
    sessions: list[Session], seed: int = 1729, train_fraction: float = 0.7, validation_fraction: float = 0.15
) -> dict[str, list[Session]]:
    if len(sessions) < 10:
        raise DataError("at least ten sessions are required for non-empty train, validation, and test splits")
    session_ids = [session.session_id for session in sessions]
    if len(set(session_ids)) != len(session_ids):
        raise DataError("session IDs must be unique to prevent split leakage")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise DataError("train and validation fractions must be positive and sum to less than one")
    indices = list(range(len(sessions)))
    train_count = round(len(sessions) * train_fraction)
    validation_count = round(len(sessions) * validation_fraction)
    if min(train_count, validation_count, len(sessions) - train_count - validation_count) < 1:
        raise DataError("split fractions produce an empty partition for this dataset size")
    train_idx, remainder_idx = train_test_split(indices, train_size=train_count, random_state=seed, shuffle=True)
    val_idx, test_idx = train_test_split(
        remainder_idx, train_size=validation_count, random_state=seed + 1, shuffle=True
    )
    return {
        "train": [sessions[i] for i in train_idx],
        "validation": [sessions[i] for i in val_idx],
        "test": [sessions[i] for i in test_idx],
    }


def save_sessions(splits: dict[str, list[Session]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for split_name, sessions in splits.items():
        path = directory / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for session in sessions:
                stream.write(json.dumps({"session_id": session.session_id, "items": session.items, "pattern": session.pattern}) + "\n")
    (directory / "catalogue.json").write_text(json.dumps(catalogue_rows(), indent=2) + "\n", encoding="utf-8")


def load_sessions(path: Path) -> list[Session]:
    sessions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            session = Session(str(row["session_id"]), tuple(row["items"]), str(row["pattern"]))
        except (KeyError, TypeError) as error:
            raise DataError(f"invalid session row {line_number}") from error
        if not 2 <= len(session.items) <= 8 or any(item not in product_lookup() for item in session.items):
            raise DataError(f"invalid product or basket size in row {line_number}")
        sessions.append(session)
    return sessions


def describe_sessions(sessions: Iterable[Session]) -> dict[str, object]:
    sessions = list(sessions)
    lengths = Counter(len(session.items) for session in sessions)
    frequencies = Counter(item for session in sessions for item in session.items)
    pairs = Counter(pair for session in sessions for pair in combinations(sorted(set(session.items)), 2))
    itemsets = Counter(
        itemset
        for session in sessions
        for size in (2, 3)
        if len(set(session.items)) >= size
        for itemset in combinations(sorted(set(session.items)), size)
    )
    top_pairs = [{"items": list(pair), "count": count} for pair, count in pairs.most_common(10)]
    top_itemsets = [{"items": list(itemset), "count": count} for itemset, count in itemsets.most_common(15)]
    return {
        "transactions": len(sessions),
        "unique_products": len({item for session in sessions for item in session.items}),
        "average_basket_length": round(sum(len(session.items) for session in sessions) / len(sessions), 3) if sessions else 0,
        "basket_length_distribution": dict(sorted(lengths.items())),
        "product_frequency": dict(sorted(frequencies.items())),
        "most_common_pairs": top_pairs,
        "most_common_itemsets": top_itemsets,
    }


def statistics_frame(sessions: Iterable[Session]) -> pd.DataFrame:
    return pd.DataFrame([asdict(session) | {"items": ",".join(session.items), "basket_length": len(session.items)} for session in sessions])
