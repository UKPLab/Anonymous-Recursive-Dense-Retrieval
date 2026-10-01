(function () {
  var store = (window.__rdrData = window.__rdrData || {});
  var orig = window.fetch ? window.fetch.bind(window) : null;
  function keyOf(url) {
    var m = /\/(data\/[^?#]+\.json)(?:[?#]|$)/.exec(url);
    return m ? decodeURIComponent(m[1]) : null;
  }
  function respond(key) {
    if (!(key in store)) return new Response("", { status: 404 });
    return new Response(JSON.stringify(store[key]), { status: 200, headers: { "Content-Type": "application/json" } });
  }
  window.fetch = function (input, init) {
    var url = typeof input === "string" ? input : input && input.url;
    var key = url ? keyOf(url) : null;
    if (!key) return orig ? orig(input, init) : Promise.reject(new Error("fetch unavailable"));
    if (key in store) return Promise.resolve(respond(key));
    return new Promise(function (resolve) {
      var s = document.createElement("script");
      s.src = url.replace(/([?#].*)?$/, "") + ".js";
      s.onload = function () { resolve(respond(key)); };
      s.onerror = function () { resolve(new Response("", { status: 404 })); };
      document.head.appendChild(s);
    });
  };
})();
