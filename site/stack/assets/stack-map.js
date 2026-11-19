// Draws the shared layer map into every element with class "stack-map": three groups,
// the tensor network libraries, the Rust numerical stack they use (each library usable on
// its own), and the threadteam substrate under any Rust HPC code.
// data-focus="users" or "backend" dims the layers that page does not cover.
(function () {
  var GH = "https://github.com/tensor4all/";
  // Every row is one layer named by its role; every box is one library,
  // titled with its name and linked to its GitHub repository.
  // star: true marks the flagship projects.
  var T4A = GH + "tensor4all-rs/tree/main/crates/";
  var ROWS = [
    { id: "front", kind: "user", label: "Frontends", sub: "Julia, C, Python", boxes: [
      { t: "Tensor4all.jl", d: "Julia, best effort", href: GH + "Tensor4all.jl" },
      { t: "tensor4all-capi", d: "C API over tensor4all-rs", href: T4A + "tensor4all-capi" },
      { t: "tensor4all-py", d: "PyO3 prototype", href: T4A + "tensor4all-py" } ] },
    { id: "algo", kind: "user", label: "Algorithms", sub: "TT, TCI, tree networks", boxes: [
      { t: "tensor4all-rs", d: "QTT, TCI, tree tensor networks", href: GH + "tensor4all-rs", star: true },
      { t: "quanticsgrids-rs", d: "quantics grids; no dependencies", href: GH + "quanticsgrids-rs" } ] },
    { id: "tenferro", kind: "backend", label: "Tensor runtime", sub: "tensors, AD, backends", boxes: [
      { t: "tenferro-rs", d: "tensors, autodiff, CPU, CUDA, XLA", href: GH + "tenferro-rs", star: true } ] },
    { id: "prov", kind: "backend", label: "Kernel providers", sub: "contraction, linalg", boxes: [
      { t: "cpueinsum-rs", d: "N-ary einsum, given order", href: GH + "cpueinsum-rs" },
      { t: "tcontract-rs", d: "contraction, TAPP C ABI", href: GH + "tcontract-rs" },
      { t: "tlinalg-rs", d: "batched SVD, QR, eigh, LU", href: GH + "tlinalg-rs" } ] },
    { id: "found", kind: "backend", label: "Foundation", sub: "views, dense linalg", boxes: [
      { t: "strided-rs", d: "strided views", href: GH + "strided-rs" },
      { t: "colmat-rs", d: "linalg, from faer", href: GH + "colmat-rs", star: true } ] },
    // wide: the band runs under hataori too, because every CPU thread in the
    // stack, and every thread a hataori lane computes on, comes from threadteam.
    { id: "threads", kind: "neutral", label: "Threads", sub: "OpenMP-style teams", wide: true, boxes: [
      { t: "threadteam-rs", d: "the one owner of CPU threads: the stack, hataori and host codes (DFT, quantum chemistry) all run on its teams", href: GH + "threadteam-rs" } ] }
  ];
  var NS = "http://www.w3.org/2000/svg";
  var W = 960, BAND_X = 16, BAND_W = 704, ROW_H = 72, GAP = 16, TOP = 16;
  var LABEL_H = 30, SPLIT = 2, SPLIT_GAP = 64, SPLIT2 = 5, SPLIT2_GAP = 28;  // rows before SPLIT are the tensor network libraries
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
    function rowY(i) { return TOP + LABEL_H + i * (ROW_H + GAP) + (i >= SPLIT ? SPLIT_GAP + LABEL_H : 0) + (i >= SPLIT2 ? SPLIT2_GAP + LABEL_H : 0); }
    var H = rowY(ROWS.length - 1) + ROW_H + TOP + 8;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Layer map in three groups. Top: tensor network libraries (frontends, tensor4all-rs, quanticsgrids-rs), which use the stack; tensor4all-rs depends on tenferro-rs. Middle: the Rust numerical stack (tenferro-rs, cpueinsum-rs, tcontract-rs, tlinalg-rs, strided-rs, colmat-rs), each usable on its own. Bottom: threadteam-rs, the thread substrate for any Rust HPC code, which the stack and hataori run on. hataori sits beside and lends execution resources. The flagship projects tensor4all-rs, tenferro-rs and colmat-rs are starred." });
    var title = el("title", {}, svg);
    title.textContent = "tensor4all Rust stack";

    function frame(i0, i1, label, cls, x1) {
      var y0 = rowY(i0) - LABEL_H - 4, y1 = rowY(i1) + ROW_H + 8;
      el("rect", { x: BAND_X - 8, y: y0, width: x1 - BAND_X + 16, height: y1 - y0, rx: 14, "class": "sm-group " + cls }, svg);
      text(svg, BAND_X + 6, y0 + 21, "sm-grouplabel", label);
    }
    frame(0, SPLIT - 1, "Tensor network libraries: users of the stack", "user", BAND_X + BAND_W);
    frame(SPLIT, SPLIT2 - 1, "Rust numerical stack: each library usable on its own", "backend", BAND_X + BAND_W);
    frame(SPLIT2, ROWS.length - 1, "Thread substrate: for any Rust HPC code, not only this stack", "neutral", SIDE_X + SIDE_W);
    ROWS.forEach(function (row, i) {
      var y = rowY(i);
      var dim = (focus === "users" && row.kind !== "user" && row.id !== "tenferro") ||
                (focus === "backend" && row.kind === "user");
      var g = el("g", { "class": dim ? "sm-dim" : "" }, svg);
      var bandW = row.wide ? SIDE_X + SIDE_W - BAND_X : BAND_W;
      var boxX1 = row.wide ? SIDE_X + SIDE_W - 12 : BOX_X1;
      el("rect", { x: BAND_X, y: y, width: bandW, height: ROW_H, rx: 10, "class": "sm-band " + row.kind }, g);
      text(g, BAND_X + 16, y + 32, "sm-label", row.label);
      text(g, BAND_X + 16, y + 50, "sm-sub", row.sub);
      var n = row.boxes.length;
      var bw = (boxX1 - BOX_X0 - (n - 1) * BOX_GAP) / n;
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

    var ux = BOX_X0 + 250, uy0 = rowY(SPLIT - 1) + ROW_H - 11, uy1 = rowY(SPLIT) + 11;
    el("path", { d: "M" + ux + "," + uy0 + " L" + ux + "," + (uy1 - 14), "class": "sm-arrow" }, svg);
    el("polygon", { points: ux + "," + uy1 + " " + (ux - 7) + "," + (uy1 - 14) + " " + (ux + 7) + "," + (uy1 - 14), "class": "sm-head" }, svg);
    text(svg, ux + 150, (uy0 + uy1) / 2 + 4, "sm-verb", "depends on tenferro-rs, like any other user");
    var tx = BOX_X0 + 250, ty0 = rowY(SPLIT2 - 1) + ROW_H + 9, ty1 = rowY(SPLIT2) + 11;
    el("path", { d: "M" + tx + "," + ty0 + " L" + tx + "," + (ty1 - 14), "class": "sm-arrow" }, svg);
    el("polygon", { points: tx + "," + ty1 + " " + (tx - 7) + "," + (ty1 - 14) + " " + (tx + 7) + "," + (ty1 - 14), "class": "sm-head" }, svg);
    text(svg, tx + 110, ty0 + 22, "sm-verb", "takes its threads from threadteam");
    // hataori beside the stack: it lends MPI ranks and thread domains to rows 1 to 3
    var sy = rowY(1), sh = rowY(3) + ROW_H - rowY(1);
    var side = el("a", { href: GH + "hataori-rs", "class": "sm-link", target: "_top" }, svg);
    el("rect", { x: SIDE_X, y: sy, width: SIDE_W, height: sh, rx: 10, "class": "sm-side" }, side);
    text(side, SIDE_X + 14, sy + 28, "sm-sidetitle", "hataori-rs");
    ["map and pmap over", "serial, Rayon, MPI", "", "splits the machine", "into MPI ranks and", "thread domains for", "outer parallel runs"].forEach(function (s, k) {
      if (s) text(side, SIDE_X + 14, sy + 52 + k * 17, "sm-desc", s);
    });
    var tipX = BAND_X + BAND_W + 2;
    for (var r = 1; r <= 3; r++) {
      var cy = rowY(r) + ROW_H / 2;
      el("path", { d: "M" + SIDE_X + "," + cy + " L" + (tipX + 12) + "," + cy, "class": "sm-arrow" }, svg);
      el("polygon", { points: tipX + "," + cy + " " + (tipX + 14) + "," + (cy - 7) + " " + (tipX + 14) + "," + (cy + 7),
        "class": "sm-head" }, svg);
      text(svg, (tipX + SIDE_X) / 2 + 6, cy - 8, "sm-verb", "lends");
    }
    // hataori itself sits on threadteam: its lanes lead threadteam teams
    var ax = SIDE_X + SIDE_W / 2, ay0 = sy + sh;
    var ay1 = rowY(ROWS.length - 1) - 2;
    el("path", { d: "M" + ax + "," + ay0 + " L" + ax + "," + (ay1 - 12), "class": "sm-arrow" }, svg);
    el("polygon", { points: ax + "," + ay1 + " " + (ax - 7) + "," + (ay1 - 14) + " " + (ax + 7) + "," + (ay1 - 14),
      "class": "sm-head" }, svg);
    text(svg, ax + 36, (ay0 + ay1) / 2 + 4, "sm-verb", "runs on");
    host.appendChild(svg);
  }

  document.querySelectorAll(".stack-map").forEach(draw);
})();
