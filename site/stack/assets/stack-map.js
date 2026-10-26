// Draws the shared layer map into every element with class "stack-map".
// data-focus="users" or "backend" dims the layers that page does not cover.
(function () {
  var GH = "https://github.com/tensor4all/";
  // Every row is one layer named by its role; every box is one library,
  // titled with its name and linked to its GitHub repository.
  var T4A = GH + "tensor4all-rs/tree/main/crates/";
  var ROWS = [
    { id: "front", kind: "user", label: "Frontends", sub: "Julia, C, Python", boxes: [
      { t: "Tensor4all.jl", d: "Julia frontend, best effort", href: GH + "Tensor4all.jl" },
      { t: "tensor4all-capi", d: "C API over tensor4all-rs", href: T4A + "tensor4all-capi" },
      { t: "tensor4all-py", d: "PyO3 prototype", href: T4A + "tensor4all-py" } ] },
    { id: "algo", kind: "user", label: "Algorithms", sub: "TT, TCI, tree networks", boxes: [
      { t: "tensor4all-rs", d: "TT and TCI stack, network stack", href: GH + "tensor4all-rs" },
      { t: "quanticsgrids-rs", d: "quantics grids, coordinates", href: GH + "quanticsgrids-rs" } ] },
    { id: "tenferro", kind: "backend", label: "Tensor runtime", sub: "tensors, AD, backends", boxes: [
      { t: "tenferro-rs", d: "tensor types, autodiff, CPU/GPU/XLA", href: GH + "tenferro-rs" },
      { t: "computegraph-rs", d: "graph IR for traced AD", href: GH + "computegraph-rs" } ] },
    { id: "prov", kind: "backend", label: "Kernel providers", sub: "contraction, linalg", boxes: [
      { t: "cpueinsum-rs", d: "N-ary einsum, given order", href: GH + "cpueinsum-rs" },
      { t: "tcontract-rs", d: "binary contraction, C ABI", href: GH + "tcontract-rs" },
      { t: "tlinalg-rs", d: "batched SVD, QR, eigh, LU", href: GH + "tlinalg-rs" } ] },
    { id: "found", kind: "backend", label: "Foundation", sub: "views, dense kernels", boxes: [
      { t: "strided-rs", d: "shared strided view types", href: GH + "strided-rs" },
      { t: "t4a-faer", d: "our fork of faer, pure Rust", href: GH + "faer" },
      { t: "BLAS, LAPACK", d: "vendor library, optional", href: null } ] }
  ];
  var NS = "http://www.w3.org/2000/svg";
  var W = 960, BAND_X = 16, BAND_W = 736, ROW_H = 72, GAP = 16, TOP = 16;
  var BOX_X0 = 176, BOX_X1 = 740, BOX_GAP = 12;

  function el(name, attrs, parent) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function text(parent, x, y, cls, s) {
    var t = el("text", { x: x, y: y, "class": cls }, parent);
    t.textContent = s;
    return t;
  }

  function draw(host) {
    var focus = host.getAttribute("data-focus") || "all";
    var H = TOP + ROWS.length * ROW_H + (ROWS.length - 1) * GAP + TOP;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Layer map of the tensor4all Rust stack, from frontends down to shared foundations, with hataori beside it" });
    var title = el("title", {}, svg);
    title.textContent = "tensor4all Rust stack";
    var defs = el("defs", {}, svg);
    var m = el("marker", { id: "sm-arrow-" + focus, viewBox: "0 0 10 10", refX: "9", refY: "5",
      markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", "class": "sm-head" }, m);

    ROWS.forEach(function (row, i) {
      var y = TOP + i * (ROW_H + GAP);
      var dim = (focus === "users" && row.kind === "backend" && row.id !== "tenferro") ||
                (focus === "backend" && row.kind === "user");
      var g = el("g", { "class": dim ? "sm-dim" : "" }, svg);
      el("rect", { x: BAND_X, y: y, width: BAND_W, height: ROW_H, rx: 10, "class": "sm-band " + row.kind }, g);
      text(g, BAND_X + 16, y + 32, "sm-label", row.label);
      text(g, BAND_X + 16, y + 50, "sm-sub", row.sub);
      var n = row.boxes.length;
      var bw = (BOX_X1 - BOX_X0 - (n - 1) * BOX_GAP) / n;
      row.boxes.forEach(function (b, j) {
        var x = BOX_X0 + j * (bw + BOX_GAP);
        var parent = g;
        if (b.href) {
          parent = el("a", { href: b.href, "class": "sm-link", target: "_top" }, g);
        }
        el("rect", { x: x, y: y + 11, width: bw, height: ROW_H - 22, rx: 7, "class": "sm-box" }, parent);
        text(parent, x + 12, y + 32, "sm-title", b.t);
        text(parent, x + 12, y + 49, "sm-desc", b.d);
      });
    });

    // hataori beside the stack, feeding execution resources into rows 1 to 3
    var sx = 768, sw = 176, sy = TOP + (ROW_H + GAP), sh = 3 * ROW_H + 2 * GAP;
    var side = el("a", { href: GH + "hataori-rs", "class": "sm-link", target: "_top" }, svg);
    el("rect", { x: sx, y: sy, width: sw, height: sh, rx: 10, "class": "sm-side" }, side);
    text(side, sx + 14, sy + 28, "sm-sidetitle", "hataori-rs");
    ["map and pmap over", "serial, Rayon, MPI", "", "hands thread pools", "and MPI ranks to", "the explicit", "execution contexts", "below"].forEach(function (s, k) {
      if (s) text(side, sx + 14, sy + 52 + k * 17, "sm-desc", s);
    });
    for (var r = 1; r <= 3; r++) {
      var cy = TOP + r * (ROW_H + GAP) + ROW_H / 2;
      el("path", { d: "M" + sx + "," + cy + " L" + (BAND_X + BAND_W + 2) + "," + cy,
        "class": "sm-arrow", "marker-end": "url(#sm-arrow-" + focus + ")" }, svg);
    }
    host.appendChild(svg);
  }

  document.querySelectorAll(".stack-map").forEach(draw);
})();
