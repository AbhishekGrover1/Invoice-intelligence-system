# Data

This project trains on **[Vendor Performance Analysis](https://www.kaggle.com/datasets/harshmadhavan/vendor-performance-analysis)**,
a public procurement/inventory dataset distributed as a single SQLite database
(`inventory.db`, ~400 MB).

## Why the database isn't in this repo

GitHub rejects pushes containing files over 100MB, and a 400MB binary has no
place in a git history regardless. More importantly, **it doesn't need to
be**: the trained model artifacts in [`/models`](../models) are the only
thing the running application (`backend/`, `inference/`) actually loads.
The database is only needed if you want to *retrain* the models yourself.

## Schema

`inventory.db` contains 5 tables:

| Table | Description | Rows |
|---|---|---|
| `vendor_invoice` | One row per vendor invoice — quantity, dollar amount, freight, PO/invoice/pay dates | 5,543 |
| `purchases` | Line-item purchase order detail — links invoices to received goods | 2,372,474 |
| `purchase_prices` | Vendor + brand price list | ~12,000 |
| `begin_inventory` | Inventory snapshot at period start | — |
| `end_inventory` | Inventory snapshot at period end | — |

Both training pipelines in this repo only touch `vendor_invoice` and
`purchases`:

- `freight_cost_prediction/data_preprocessing.py` reads `vendor_invoice` directly.
- `invoice_flagging/data_preprocessing.py` joins `vendor_invoice` to a
  per-PO aggregate of `purchases` (total received quantity/dollars per
  purchase order).

## Getting the data yourself

1. Download `inventory.db` from the [Kaggle dataset page](https://www.kaggle.com/datasets/harshmadhavan/vendor-performance-analysis).
2. Place it at `Data/inventory.db` (this exact path — both preprocessing
   modules resolve it relative to the project root).
3. Run either training script — see the [root README](../README.md#training-the-models-optional)
   for exact commands.

## What's actually in this folder

```
Data/
├── README.md                         <- you are here
└── samples/
    ├── vendor_invoice_sample.csv     <- 60 real rows, raw vendor_invoice columns
    └── invoice_risk_batch_sample.csv <- 27 real rows, pre-shaped to the invoice-risk
                                          model's exact 5-feature schema (used by the
                                          "Batch Upload" demo on the live site)
```

Both CSVs are small, real excerpts (not synthetic data) included so you can
exercise the code and the batch-upload demo without downloading the full
database.
