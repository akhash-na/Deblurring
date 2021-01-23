# DeepDeblur

```bash
pip install scikit-image
python main.py -data_folder /GOPRO_Large -batch_size 8 -save_every 10 -n_features 32 -n_resblocks 10 -n_epochs 100 -n_scales 2 -do_train true -do_test true
python main.py -data_folder /GOPRO_Large -batch_size 8 -save_every 10 -n_features 32 -n_resblocks 10 -n_epochs 100 -n_scales 2 -do_train true -do_test true -pretrained checkpoint-epoch-100.pt
```

