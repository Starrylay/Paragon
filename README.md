# Paragon

![image](https://github.com/Starrylay/Paragon/blob/main/images/main_paragon.png)

Paragon is an innovative project that combines recommendation systems with diffusion models.


# News

Accepted by Recsys 2025！

## Project Structure

The project is organized into two main components:

1. **Recommendation Model (Rec folder)**
   - Training and testing code for the recommendation model
   - Located in the `Rec` folder

2. **Diffusion Model (DIFFUSION folder)**
   - Training code for the diffusion model
   - Code to generate parameters for the recommendation model
   - Located in the `DIFFUSION` folder

## Installation

To set up the project, follow these steps:

```bash
git clone https:*******************/Paragon.git
cd Paragon
pip install -r requirements.txt
```

## Usage

### Recommendation Model

To train the recommendation model:

```bash
cd Rec
# Then find the config.ini file and change the mode inside it to train
python Rec_mian.py
```

To test a trained or generated recommendation model:

```bash
cd Rec
# Then find the config.ini file and change the mode inside it to test
python test_Recmodels.py
```

### Diffusion Model

To train the diffusion model and generate parameters for the recommendation model:

```bash
cd DIFFUSION
python main.py
```

## Contributing

We welcome contributions to Paragon! Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

