# Transformer Health Index AI

Python app jo `Transformer_HI_48_Input_Weighted_Anomaly_Dataset.xlsx` ke 48 inputs se Transformer Health Index predict karta hai.

Current behavior: workbook ke `Training_Data_48` tab se model train hota hai aur target `HI_Final` hai. Recommendation reasons `capping_with_reasons.xlsx` se load hote hain, so issue value ke saath condition aur reason show hota hai. App-level hard-cap checks DGA, BDV, moisture, IR, PI, OTI aur WTI max par apply hote hain.

## Files

- `health_rules.py` - gas threshold table, rule scores, feature builder.
- `train_model.py` - `Training_Data_48` sheet load karta hai aur 48-input NumPy neural model train karta hai.
- `app.py` - dynamic 48-parameter web index page aur prediction API.
- `model/health_index_model.npz` - training ke baad generated model.
- `model/metadata.json` - training metrics.
- `outputs/DGA_HI_Training_Set_Augmented.csv` - new dataset rows plus derived gas scores/conditions/threshold features, reason columns, Label_Rule features, `HI_before_label_rule`, and `HI_training_target`.

## Run

```powershell
cd transformer_health_index_ai
python train_model.py
python app.py
```

Open:

```text
http://127.0.0.1:8000
```

## Deploy

The app is deployment-ready with project-local data files.

- `data/Transformer_HI_48_Input_Weighted_Anomaly_Dataset.xlsx`
- `data/capping_with_reasons.xlsx`
- `requirements.txt`
- `Procfile`
- `render.yaml`

For Render/Railway-style hosting, use:

```text
Build command: pip install -r requirements.txt
Start command: python app.py
```

The app reads the hosting platform `PORT` environment variable automatically.

## Model Inputs

The model uses all 48 workbook input features from columns B:AW of `Training_Data_48`.

- DGA: H2, CH4, C2H6, C2H4, C2H2, CO, CO2
- Oil/thermal/loading/age: BDV, Moisture, WTI HV/LV, OTI, Loading, Ambient Temp, Age
- IR/PI: HV/LV/TV 1-min and 10-min IR plus PI HV/LV/TV
- Winding and bushing tan-delta/capacitance change parameters
- Training target: `HI_training_target = HI_Final`
