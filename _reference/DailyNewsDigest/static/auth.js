/**
 * Injects login / user state into #auth-header on every page.
 * Uses GET /api/me (with credentials) and renders "Login with Google" or "email | Logout".
 */
(function () {
  var el = document.getElementById("auth-header");
  if (!el) return;

  function renderLoggedOut() {
    el.innerHTML = '<a href="/login" class="auth-btn-login">Login with Google</a>';
  }

  function renderLoggedIn(data) {
    var label = data.name || data.email || "User";
    el.innerHTML =
      '<span class="auth-user">' +
      escapeHtml(label) +
      '</span> <a href="/logout" class="auth-logout">Logout</a>';
  }

  function escapeHtml(s) {
    var div = document.createElement("div");
    div.textContent = s == null ? "" : s;
    return div.innerHTML;
  }

  fetch("/api/me", { credentials: "same-origin" })
    .then(function (r) {
      return r.json().then(function (data) {
        if (data && data.logged_in) {
          renderLoggedIn(data);
        } else {
          renderLoggedOut();
        }
        var navSettings = document.getElementById("nav-settings");
        if (navSettings) {
          navSettings.style.display = (data && data.logged_in) ? "" : "none";
        }
      });
    })
    .catch(function () {
      renderLoggedOut();
      var navSettings = document.getElementById("nav-settings");
      if (navSettings) navSettings.style.display = "none";
    });
})();
