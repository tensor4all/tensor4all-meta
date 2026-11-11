// Draws the shared layer map into every element with class "stack-map".
// data-focus="users" or "backend" dims the layers that page does not cover.
(function () {
  var GH = "https://github.com/tensor4all/";
  // Every row is one layer named by its role; every box is one library,
  // titled with its name and linked to its GitHub repository.
  // star: true marks the two flagship projects.
  var T4A = GH + "tensor4all-rs/tree/main/crates/";
  var ROWS = [
    { id: "front", kind: "user", label: "Frontends", sub: "Julia, C, Python", boxes: [
      { t: "Tensor4all.jl", d: "Julia, best effort", href: GH + "Tensor4all.jl" },
      { t: "tensor4all-capi", d: "C API over tensor4all-rs", href: T4A + "tensor4all-capi" },
      { t: "tensor4all-py", d: "PyO3 prototype", href: T4A + "tensor4all-py" } ] },
    { id: "algo", kind: "user", label: "Algorithms", sub: "TT, TCI, tree networks", boxes: [
      { t: "tensor4all-rs", d: "QTT, TCI, tree tensor networks", href: GH + "tensor4all-rs", star: true },
      { t: "quanticsgrids-rs", d: "quantics grids, coordinates", href: GH + "quanticsgrids-rs" } ] },
    { id: "tenferro", kind: "backend", label: "Tensor runtime", sub: "tensors, AD, backends", boxes: [
      { t: "tenferro-rs", d: "tensors, autodiff, CPU, CUDA, XLA", href: GH + "tenferro-rs", star: true },
      { t: "computegraph-rs", d: "graph IR for traced AD", href: GH + "computegraph-rs" } ] },
    { id: "prov", kind: "backend", label: "Kernel providers", sub: "contraction, linalg", boxes: [
      { t: "cpueinsum-rs", d: "N-ary einsum, given order", href: GH + "cpueinsum-rs" },
      { t: "tcontract-rs", d: "contraction, TAPP C ABI", href: GH + "tcontract-rs" },
      { t: "tlinalg-rs", d: "batched SVD, QR, eigh, LU", href: GH + "tlinalg-rs" } ] },
    { id: "found", kind: "backend", label: "Foundation", sub: "views, threads, linalg", boxes: [
      { t: "strided-rs", d: "strided views", href: GH + "strided-rs" },
      { t: "threadteam-rs", d: "CPU thread teams", href: GH + "threadteam-rs" },
      { t: "t4a-faer", d: "our faer fork", href: GH + "faer" },
      { t: "BLAS, LAPACK", d: "vendor, optional", href: null } ] }
  ];
  var NS = "http://www.w3.org/2000/svg";
  var W = 960, BAND_X = 16, BAND_W = 704, ROW_H = 72, GAP = 16, TOP = 16;
  var BOX_X0 = 176, BOX_X1 = 708, BOX_GAP = 12;
  var SIDE_X = 800, SIDE_W = 144;

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
      "aria-label": "Layer map of the tensor4all Rust stack, from frontends down to shared foundations, with hataori beside it. The flagship projects tensor4all-rs and tenferro-rs are starred." });
    var title = el("title", {}, svg);
    title.textContent = "tensor4all Rust stack";

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
        el("rect", { x: x, y: y + 11, width: bw, height: ROW_H - 22, rx: 7, "class": "sm-box" + (b.star ? " star" : "") }, parent);
        var t = el("text", { x: x + 12, y: y + 32, "class": "sm-title" }, parent);
        if (b.star) {
          var s = el("tspan", { "class": "sm-star" }, t);
          s.textContent = "★ ";
        }
        var name = el("tspan", {}, t);
        name.textContent = b.t;
        text(parent, x + 12, y + 49, "sm-desc", b.d);
      });
    });

    // hataori beside the stack: it lends MPI ranks and thread domains to rows 1 to 3
    var sy = TOP + (ROW_H + GAP), sh = 3 * ROW_H + 2 * GAP;
    var side = el("a", { href: GH + "hataori-rs", "class": "sm-link", target: "_top" }, svg);
    el("rect", { x: SIDE_X, y: sy, width: SIDE_W, height: sh, rx: 10, "class": "sm-side" }, side);
    text(side, SIDE_X + 14, sy + 28, "sm-sidetitle", "hataori-rs");
    ["map and pmap over", "serial, Rayon, MPI", "", "splits the machine", "into MPI ranks and", "thread domains for", "outer parallel runs"].forEach(function (s, k) {
      if (s) text(side, SIDE_X + 14, sy + 52 + k * 17, "sm-desc", s);
    });
    var tipX = BAND_X + BAND_W + 2;
    for (var r = 1; r <= 3; r++) {
      var cy = TOP + r * (ROW_H + GAP) + ROW_H / 2;
      el("path", { d: "M" + SIDE_X + "," + cy + " L" + (tipX + 12) + "," + cy, "class": "sm-arrow" }, svg);
      el("polygon", { points: tipX + "," + cy + " " + (tipX + 14) + "," + (cy - 7) + " " + (tipX + 14) + "," + (cy + 7),
        "class": "sm-head" }, svg);
      text(svg, (tipX + SIDE_X) / 2 + 6, cy - 8, "sm-verb", "lends");
    }
    host.appendChild(svg);
  }

  document.querySelectorAll(".stack-map").forEach(draw);
})();
