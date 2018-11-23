window.onload = function() {
  var demo = document.createElement("IFRAME");
  demo.src = "https://repl.it/@szha/gluon-nlp?lite=true";
  demo.height = "400px";
  demo.width = "100%";
  demo.scrolling = "no";
  demo.frameborder = "no";
  demo.allowtransparency = true;
  demo.allowfullscreen = true;
  demo.seamless = true;
  demo.sandbox = "allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-modals";
  demo_div = document.getElementById("frontpage-demo");
  while (demo_div.firstChild) {
    demo_div.removeChild(demo_div.firstChild);
  }
  demo_div.appendChild(demo);
}; // load demo last
