#!/usr/bin/env python3
"""Cifra las páginas HTML de los recursos con una clave común.

Cada página se guarda cifrada (AES-256-GCM; la clave se deriva con PBKDF2-SHA256, 600 000 iteraciones).
Lo que se publica es una «puerta» que pide la clave y descifra la página en el navegador con WebCrypto.
Sin la clave, el contenido no se puede leer ni en la web ni en el repositorio.

Las fuentes en claro NO se guardan en el repositorio: se conservan aparte.

Uso:
    python3 herramientas/cifrar_paginas.py --fuentes RUTA_FUENTES --salida . --clave "la-clave"

Cifra todos los .html de RUTA_FUENTES (salvo carpetas y archivos que empiezan por «_») y los escribe
en la misma ruta relativa dentro de --salida. La puerta muestra el logo de assets/logo_ubu.png.
Para cambiar la clave de curso, se vuelve a ejecutar con la clave nueva y se suben los archivos.
Requiere: pip install cryptography
"""
import argparse, base64, html, json, os, re, secrets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ITERACIONES = 600_000

PUERTA = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>__TITULO__</title>
<style>
  :root{--paper:#fff;--desk:#eceae6;--ink:#1b1b1b;--soft:#555;--line:#cfccc6;--red:#c00000;--red-dark:#8f1d1d;--info:#2e5f8a;color-scheme:light}
  *{box-sizing:border-box}
  body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:var(--desk);color:var(--ink);
       font-family:'Source Sans 3','Segoe UI',Calibri,Arial,sans-serif;font-size:16px;line-height:1.5;padding:24px 16px}
  .box{width:100%;max-width:420px;background:var(--paper);padding:30px 28px 24px;box-shadow:0 1px 3px rgba(0,0,0,.12);border-top:4px solid var(--red)}
  .logo{display:block;width:100%;max-width:300px;height:auto}
  .prof{margin:10px 0 0;font-size:14px;font-weight:700;color:var(--ink)}
  h1{color:var(--red);font-size:20px;line-height:1.25;margin:22px 0 4px;text-wrap:balance}
  p{margin:0 0 16px;color:var(--soft);font-size:15px}
  label{display:block;font-weight:700;margin-bottom:6px}
  input[type=password]{width:100%;font:inherit;font-size:17px;padding:11px 12px;border:1px solid var(--line);border-radius:6px;color:var(--ink);background:#fff}
  input[type=password]:focus{outline:3px solid var(--info);outline-offset:1px;border-color:var(--info)}
  .rec{display:flex;align-items:center;gap:8px;margin:12px 0 18px;font-size:15px;font-weight:400}
  .rec input{width:18px;height:18px;accent-color:var(--red)}
  button{width:100%;font:inherit;font-weight:700;font-size:17px;color:#fff;background:var(--red);border:0;border-radius:6px;padding:12px;min-height:48px;cursor:pointer}
  button:hover{background:var(--red-dark)}
  button:disabled{opacity:.6;cursor:wait}
  button:focus-visible{outline:3px solid var(--info);outline-offset:2px}
  .err{color:var(--red);font-weight:700;min-height:1.5em;margin:10px 0 0;font-size:15px}
  .pie{margin:18px 0 0;font-size:13px}
</style>
</head>
<body>
<main class="box">
  <img class="logo" src="__RAIZ__assets/logo_ubu.png" alt="Universidad de Burgos" width="640" height="244">
  <p class="prof">Prof. Kerly Palacios</p>
  <h1>__TITULO__</h1>
  <p>Panificación y Pastelería (9057). Acceso con la clave publicada en UBUVirtual.</p>
  <form id="f" autocomplete="on">
    <label for="clave">Clave de acceso</label>
    <input type="password" id="clave" name="password" autocomplete="current-password" required>
    <label class="rec" for="rec"><input type="checkbox" id="rec">Recordar en este navegador</label>
    <button type="submit" id="btn">Entrar</button>
    <p class="err" id="err" role="alert"></p>
  </form>
  <p class="pie">Si no tenéis la clave, consultad UBUVirtual o al profesorado.</p>
  <noscript><p class="err">Esta página necesita JavaScript activado.</p></noscript>
</main>
<script>
(function(){
  "use strict";
  const D = __DATOS__;
  const CLAVE = "ubu9057-clave";
  const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  const leer = st => { try { return st.getItem(CLAVE); } catch (_) { return null; } };
  const guardar = (st, v) => { try { st.setItem(CLAVE, v); } catch (_) {} };
  const borrar = st => { try { st.removeItem(CLAVE); } catch (_) {} };
  async function descifrar(pw) {
    const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(pw), "PBKDF2", false, ["deriveKey"]);
    const k = await crypto.subtle.deriveKey({name: "PBKDF2", salt: b64(D.s), iterations: D.i, hash: "SHA-256"},
                                            base, {name: "AES-GCM", length: 256}, false, ["decrypt"]);
    return new TextDecoder().decode(await crypto.subtle.decrypt({name: "AES-GCM", iv: b64(D.n)}, k, b64(D.c)));
  }
  function mostrar(doc) { document.open(); document.write(doc); document.close(); }
  const f = document.getElementById("f"), btn = document.getElementById("btn"), err = document.getElementById("err");
  if (!window.crypto || !crypto.subtle) { err.textContent = "Este navegador no puede abrir la página. Actualizadlo o usad otro."; btn.disabled = true; return; }
  f.addEventListener("submit", async e => {
    e.preventDefault(); err.textContent = ""; btn.disabled = true; btn.textContent = "Abriendo…";
    const pw = document.getElementById("clave").value;
    try {
      const doc = await descifrar(pw);
      guardar(sessionStorage, pw);
      if (document.getElementById("rec").checked) guardar(localStorage, pw); else borrar(localStorage);
      mostrar(doc);
    } catch (_) {
      err.textContent = "Clave incorrecta. Revisadla y volved a intentarlo.";
      btn.disabled = false; btn.textContent = "Entrar";
    }
  });
  const guardada = leer(sessionStorage) || leer(localStorage);
  if (guardada) {
    btn.disabled = true; btn.textContent = "Abriendo…";
    descifrar(guardada).then(mostrar).catch(() => {
      borrar(sessionStorage); borrar(localStorage);
      btn.disabled = false; btn.textContent = "Entrar";
      err.textContent = "La clave guardada ya no es válida. Escribid la clave vigente.";
    });
  }
})();
</script>
</body>
</html>
"""


def cifrar(texto, clave):
    sal, nonce = secrets.token_bytes(16), secrets.token_bytes(12)
    k = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=sal, iterations=ITERACIONES).derive(clave.encode("utf-8"))
    ct = AESGCM(k).encrypt(nonce, texto.encode("utf-8"), None)
    e = lambda b: base64.b64encode(b).decode("ascii")
    return {"v": 1, "i": ITERACIONES, "s": e(sal), "n": e(nonce), "c": e(ct)}


TEMAS = {"rojo": ("#c00000", "#8f1d1d"), "azul": ("#1d4f91", "#133a6c")}


def puerta(texto, clave, raiz="", tema="rojo"):
    m = re.search(r"<title>(.*?)</title>", texto, re.S)
    titulo = html.escape(html.unescape(m.group(1).strip())) if m else "Recurso de la asignatura"
    datos = json.dumps(cifrar(texto, clave), separators=(",", ":"))
    p, o = TEMAS[tema]
    html_ = PUERTA.replace("--red:#c00000;--red-dark:#8f1d1d;", f"--red:{p};--red-dark:{o};")
    return html_.replace("__TITULO__", titulo).replace("__RAIZ__", raiz).replace("__DATOS__", datos)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuentes", required=True, help="carpeta con las páginas en claro (fuera del repositorio)")
    ap.add_argument("--salida", required=True, help="carpeta del repositorio")
    ap.add_argument("--clave", required=True, help="clave común de acceso")
    ap.add_argument("--tema", choices=sorted(TEMAS), default="rojo", help="color de la puerta: rojo (docencia) o azul (prácticas)")
    a = ap.parse_args()
    n = 0
    for raiz, carpetas, archivos in os.walk(a.fuentes):
        carpetas[:] = [c for c in carpetas if not c.startswith(("_", "."))]   # _base, etc.: uso interno
        for nom in sorted(archivos):
            if not nom.endswith(".html") or nom.startswith("_"):
                continue
            rel = os.path.relpath(os.path.join(raiz, nom), a.fuentes)
            dest = os.path.join(a.salida, rel)
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            with open(os.path.join(raiz, nom), encoding="utf-8") as f:
                texto = f.read()
            with open(dest, "w", encoding="utf-8") as f:
                f.write(puerta(texto, a.clave, "../" * rel.replace(os.sep, "/").count("/"), a.tema))
            print("cifrada:", rel); n += 1
    print(f"{n} páginas cifradas.")


if __name__ == "__main__":
    main()
