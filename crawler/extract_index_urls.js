function extract() {

  var imgs_src = [];

  var imgs = document.querySelectorAll('img#img');
  for (img in imgs) {
    if (imgs[img].src) {
      href = imgs[img].parentElement.href;
      if (!(imgs_src.includes(href))) {
        imgs_src.push(href);
      }
    }
  }

  return imgs_src;
};

var data = extract();

return data;
