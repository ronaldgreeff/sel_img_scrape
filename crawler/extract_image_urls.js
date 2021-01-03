function extract() {

  l = [];

  var imgs = document.querySelectorAll("#imgAttach");
  for (img in imgs) {
    if (imgs[img].src) {
      l.push(imgs[img].src);
    }
  }
  return l;
};

var data = extract();

return data;
