  def demo_digit(folder=pathlib.Path.cwd(), *args, **kwargs):
      if isinstance(folder, str):
          folder = pathlib.Path(folder)
      from sklearn import datasets
      model = FastICA(*args, **kwargs)
      ip = ImageEncoder(model, size=(8,8))
      digists = datasets.load_digits()
      ip._fit(digists.data * (255//16))

      save_folder = folder / 'eigen'
      save_folder.mkdir(exist_ok=True)
      for k, im in enumerate(ip.eigen_images):
          im.save(save_folder / f'{k}.jpg')
      # generate artificial images
      for k, im in enumerate(ip.generate(10, toimage=True)):
          im.save(save_folder / f'artificial{k}.jpg')


  demo_digit(n_components=15)
