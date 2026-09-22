# Kaip paleisti šį darbą (Google Colab)

Ši instrukcija skirta žmogui, kuris **nebūtinai programuoja**. Darykite punktus **iš eilės**. Visi penki notebook’ai leidžiami per [Google Colab](https://colab.research.google.com/).

Kodėl Colab, o ne lokali naršyklės Python aplinka, parašyta **6 punkte**.

---

## 1. Paleiskite notebook’us 01–04 per Google Colab

Kiekvieną iš keturių notebook’ų paleiskite **atskiroje** Colab sesijoje (naujas skirtukas / naujas notebook).

Kartojate šiuos žingsnius **keturis kartus**: `01_baseline_logistine_regresija.ipynb`, tada `02_catboost.ipynb`, tada `03_svm.ipynb`, tada `04_mlp.ipynb`.

1. Atidarykite naršyklėje: [https://colab.research.google.com](https://colab.research.google.com).
2. Meniu **File → Upload notebook** (Failas → Įkelti notebook).
3. Pasirinkite norimą `.ipynb` failą iš šio projekto aplanko.
4. Kairėje pusėje atidarykite aplanko piktogramą (**Files** / Failai).
5. Paspauskite **Upload** ir **įkelkite `bank-full.csv`**.  
   Svarbu: šį failą reikia įkelti **į kiekvieną naują Colab sesiją atskirai** (kitas notebook = naujas įkėlimas).
6. Meniu **Runtime → Run all** (Vykdyti viską). Palaukite, kol visos celės baigs darbą.  
   `02_catboost` pirmą kartą gali užtrukti ilgiau (įdiegia CatBoost).

---

## 2. Paskutinė celė pati atsiunčia ZIP į kompiuterį

Kai notebook’as nueina iki galo, **paskutinė celė** sukuria ZIP ir jį atsisiunčia.

Failas atsiras jūsų kompiuterio aplanke **Atsisiuntimai (Downloads)**.

Pavyzdžiai, ką pamatysite:

- `baseline_outputs.zip`
- `catboost_outputs.zip`
- `svm_outputs.zip`
- `mlp_outputs.zip`

Jei naršyklė klausia, ar leisti atsisiuntimą — paspauskite **Allow / Leisti**.

---

## 3. Kiekvieną ZIP išpakuokite

1. Atidarykite aplanką **Atsisiuntimai**.
2. Ant ZIP failo spauskite **dešinį pelės klavišą**.
3. Rinkitės **Extract All…** / **Ištraukti viską** (arba Extract / Ištraukti).
4. Viduje ieškokite failo vardu `preds_<modelis>.csv`.

Po visų keturių notebook’ų turite turėti:

- `preds_baseline.csv`
- `preds_catboost.csv`
- `preds_svm.csv`
- `preds_mlp.csv`

(Kartais jie guli poaplankyje `outputs` — tada tiesiog atidarykite jį ir paimkite CSV.)

---

## 4. Surinkite visus 4 `preds_*.csv` į vieną vietą

Nukopijuokite visus keturis CSV failus į **tą patį aplanką**, kuriame yra  
`05_palyginimas_ir_rezultatai.ipynb`  
(kad būtų paprasčiau juos rasti, kai kelsite į Colab).

---

## 5. Paleiskite palyginimą (05) taip pat per Google Colab

1. Colab: **File → Upload notebook** → pasirinkite `05_palyginimas_ir_rezultatai.ipynb`.
2. Kairėje **Files** (aplankas) → **Upload**.
3. Įkelkite **visus 4** failus:  
   `preds_baseline.csv`, `preds_catboost.csv`, `preds_svm.csv`, `preds_mlp.csv`.  
   Galite pažymėti visus iš karto. Jei kompiuteris neleidžia kelių failų vienu metu — kelkite **po vieną**.
4. **Šiam 05 notebook’ui `bank-full.csv` nebūtinas palyginimo lentelei** (pakanka keturių `preds_*.csv`). Jei norite FN pavyzdžių su amžiumi, darbu ir t. t., papildomai įkelkite ir `bank-full.csv`.
5. **Runtime → Run all**.
6. Paskutinė celė atsiųs **`palyginimas_outputs.zip`** į Atsisiuntimus. Viduje bus palyginimo grafikai (`pr_all_models.png`, `calibration_all_models.png`) ir lentelė. Juos įkelkite į Word dokumentacijos 1 ir 2 pav.

Kartu galite kairėje **Files** atidaryti aplanką `outputs`, ant PNG paspausti tris taškus → **Download**.

Notebook’as skaito `preds_*.csv` iš to paties Colab aplanko, į kurį ką tik įkėlėte failus. **Google Drive čia nenaudojamas.**

---

## 6. Kodėl viskas per Colab (ypač CatBoost)

**CatBoost** yra sudėtinga biblioteka, parašyta ne grynu Python, o iš dalies **C++ kalba** (dėl greičio). Naršyklinė lokali Python aplinka yra supaprastinta versija, veikianti be tikros operacinės sistemos — ji nemoka paleisti tokių sudėtingų, ne-grynai-Python bibliotekų kaip CatBoost.

Dėl šios priežasties reikia per Colab leisti bent **`02_catboost.ipynb`**. Likusius notebook’us (01, 03, 04, 05) techniškai būtų galima leisti ir lokalioje naršyklinėje aplinkoje (jie nenaudoja CatBoost), bet šiame projekte **visi penki leidžiami per Colab**, kad nereikėtų persijunginėti tarp dviejų skirtingų aplinkų — taip paprasčiau ir nuosekliau.

---

## Trumpa atmintinė

| Eilės nr. | Ką darote |
|---|---|
| 1 | Colab → Upload notebook 01 → įkelti `bank-full.csv` → Run all → atsisiunčiamas ZIP |
| 2 | Tas pats su 02, 03, 04 (kiekvieną kartą iš naujo įkelti `bank-full.csv`) |
| 3 | ZIP išpakuoti, rasti `preds_*.csv` |
| 4 | Colab → Upload notebook 05 → įkelti visus 4 `preds_*.csv` → Run all |
