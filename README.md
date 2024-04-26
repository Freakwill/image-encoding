# image-encoding
image encoding by dimensional reduction models (in scikit-learn)

## Examples

### 1
```python
    def demo_digit(folder=pathlib.Path.cwd(), *args, **kwargs):
        if isinstance(folder, str):
            folder = pathlib.Path(folder)
        from sklearn import datasets
        model = FastICA(*args, **kwargs)   # the backend model
        enc = ImageEncoder(model, size=(8,8))
        digists = datasets.load_digits()
        ip.fit(digists.data * (255//16))

        save_in(enc.eigen_images, folder / 'eigen', exist_ok=True)
        # generate new images
        save_in(enc.generate(10, toimage=True), folder / 'eigen', exist_ok=True, prefix='generated')


    demo_digit(n_components=15)
```

### 2

```python
    def demo_face(folder=pathlib.Path.cwd(), *args, **kwargs):
        # save images in the folder before demo
        
        if isinstance(folder, str):
            folder = pathlib.Path(folder)
        # define a model, such as PCA, NMF
        model = PCA(*args, **kwargs)
        enc = ImageEncoder(model)
        # a user-friendly API calling `fit` method of the model
        enc.ezfit(folder=folder)  # folder where the images are stored
        # save the eigen images in `eigen/` subfolder
        save_in(enc.eigen_images, folder / 'eigen', exist_ok=True)
        # generate new images and save them in `generated/` subfolder
        save_in(enc.generate(10, toimage=True), folder / 'generated', exist_ok=True)

    # save the images (with the same size and mode) in the current path or a special folder
    demo_face(n_components=10)
```
