window.onload = function() {
    var demo = '<iframe src="https://repl.it/@szha/gluon-nlp?lite=true"' +
               'height="400px" width="100%" scrolling="no"' +
               'frameborder="no" allowtransparency="true" allowfullscreen="true"' +
               'sandbox="allow-forms allow-pointer-lock allow-popups allow-same-origin' +
               'allow-scripts allow-modals"></iframe>';
    var demo_div = document.getElementById("frontpage-demo");
    demo_div.innerHTML = demo;
}; // load demo last
