import React, { useEffect, useState } from "react";

const INITIAL = {
  seed: "veinn-demo-seed", message: "", in_path: "", pubfile: "public_key.json",
  out_file: "enc_pub_veinn", file_type: "json", n: 512, rounds: 15,
  layers_per_round: 15, shuffle_stride: 15, q: 2013265921, seed_len: 32,
  mode: "cbc", nonce: "", use_lwe: true, enc_file: "enc_pub_veinn.json",
  validity_window: 3600, privfile: "private_key.json", keystore: "keystore.json",
  passphrase: "", key_name: "", store_private: false,
};

const OPERATIONS = {
  encrypt: [
    ["encrypt_veinn", "VEINN seed", "Deterministic encryption using a shared seed.", "Recommended"],
    ["encrypt_public", "Recipient public key", "Hybrid encryption for a Kyber public key.", "Key required"],
  ],
  decrypt: [
    ["decrypt_veinn", "VEINN seed", "Open an artifact created with the shared seed.", "Shared seed"],
    ["decrypt_private", "Private key", "Open an artifact with a private key or keystore.", "Private material"],
  ],
  utility: [
    ["generate_keypair", "Generate keypair", "Create a Kyber public/private keypair.", "Utility"],
    ["create_keystore", "Create keystore", "Create an encrypted container for private keys.", "Utility"],
    ["derive_veinn", "Derive VEINN key", "Generate a deterministic public vector from a seed.", "Utility"],
  ],
};

function Field({ label, name, value, onChange, type = "text", hint, ...props }) {
  return <label className="field">
    <span>{label}</span>
    <input name={name} type={type} value={value} onChange={onChange} {...props} />
    {hint && <small>{hint}</small>}
  </label>;
}

function SelectField({ label, name, value, onChange, children }) {
  return <label className="field">
    <span>{label}</span>
    <select name={name} value={value} onChange={onChange}>{children}</select>
  </label>;
}

function RangeField({ label, name, value, onChange, min, max, step = 1 }) {
  return <label className="range-field">
    <span>{label}<output>{value}</output></span>
    <input name={name} type="range" value={value} min={min} max={max} step={step} onChange={onChange} />
  </label>;
}

function OperationCard({ operation, title, description, badge, active, onClick }) {
  return <button type="button" className={`operation-card ${active ? "active" : ""}`} onClick={onClick}>
    <span className="operation-radio">{active ? "●" : "○"}</span>
    <span className="operation-card-copy"><strong>{title}</strong><small>{description}</small></span>
    <em>{badge}</em>
  </button>;
}

function Log({ entries, busy }) {
  return <section className="activity-card">
    <div className="section-title"><span>Activity</span><small>LOCAL SESSION</small></div>
    <div className="activity-list">
      <p><b>system</b> Ready for a local operation.</p>
      {entries.map((entry, index) => <p className={entry.error ? "activity-error" : ""} key={`${entry.time}-${index}`}><b>{entry.time}</b> {entry.text}</p>)}
      {busy && <p className="activity-live"><b>now</b> Operation in progress…</p>}
    </div>
  </section>;
}

export default function App() {
  const [mode, setMode] = useState("encrypt");
  const [operation, setOperation] = useState("encrypt_veinn");
  const [form, setForm] = useState(INITIAL);
  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [entries, setEntries] = useState([]);
  const [selfTest, setSelfTest] = useState("running");

  useEffect(() => {
    fetch("/api/self-test")
      .then((response) => response.json().then((body) => ({ ok: response.ok, body })))
      .then(({ ok, body }) => setSelfTest(ok && body.passed ? "passed" : "failed"))
      .catch(() => setSelfTest("failed"));
  }, []);

  const update = (event) => {
    const { name, value, type, checked } = event.target;
    setForm((current) => ({ ...current, [name]: type === "checkbox" ? checked : value }));
  };

  const changeMode = (nextMode) => {
    setMode(nextMode);
    setOperation(nextMode === "encrypt" ? "encrypt_veinn" : "decrypt_veinn");
    setError("");
  };

  const submit = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    const time = new Date().toLocaleTimeString([], { hour12: false });
    setEntries((current) => [...current.slice(-5), { time, text: `${operation} started` }]);
    try {
      const payload = { ...form, operation };
      ["n", "rounds", "layers_per_round", "shuffle_stride", "q", "seed_len", "validity_window"].forEach((key) => {
        if (payload[key] !== undefined) payload[key] = Number(payload[key]);
      });
      const response = await fetch("/api/operate", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || "Operation failed.");
      if (body.output && operation.startsWith("encrypt")) setForm((current) => ({ ...current, enc_file: body.output }));
      setResult(body);
      setEntries((current) => [...current.slice(-5), { time: new Date().toLocaleTimeString([], { hour12: false }), text: body.message || "Operation complete" }]);
    } catch (submitError) {
      const message = submitError.message.replace(/\u001b\[[0-9;]*m/g, "");
      setError(message);
      setEntries((current) => [...current.slice(-5), { time: new Date().toLocaleTimeString([], { hour12: false }), text: message, error: true }]);
    } finally {
      setBusy(false);
    }
  };

  const isEncrypt = mode === "encrypt";
  const isUtility = mode === "utility";
  const isPublic = operation === "encrypt_public";
  const isPrivate = operation === "decrypt_private";
  const isKeypair = operation === "generate_keypair";
  const isKeystore = operation === "create_keystore";

  return <main className="app-shell">
    <header className="topbar">
      <div className="brand"><span className="brand-mark">V</span><span><strong>VEINN</strong><small>LOCAL CRYPTOGRAPHY</small></span></div>
      <div className={`node-status ${selfTest}`}><i /> {selfTest === "running" ? "Checking node" : selfTest === "passed" ? "Node healthy" : "Node check failed"}</div>
    </header>

    <section className="hero">
      <div><p className="eyebrow">Secure workspace</p><h1>{isUtility ? "Key management" : isEncrypt ? "Encrypt a payload" : "Decrypt a payload"}</h1><p>Choose an operation, provide only what it needs, and run it locally.</p></div>
      <div className="mode-switch" role="tablist" aria-label="Workspace mode">
        <button className={isEncrypt ? "selected" : ""} onClick={() => changeMode("encrypt")} role="tab">Encrypt</button>
        <button className={!isEncrypt && !isUtility ? "selected" : ""} onClick={() => changeMode("decrypt")} role="tab">Decrypt</button>
        <button className={isUtility ? "selected" : ""} onClick={() => { setMode("utility"); setOperation("generate_keypair"); }} role="tab">Utilities</button>
      </div>
    </section>

    <div className="workspace">
      <aside className="operation-panel">
        <div className="panel-title"><span>01</span><div><h2>Operation</h2><p>What do you want to do?</p></div></div>
        {!isUtility && <div className="operation-group">{OPERATIONS[mode].map(([key, title, description, badge]) => <OperationCard key={key} operation={key} title={title} description={description} badge={badge} active={operation === key} onClick={() => { setOperation(key); setError(""); }} />)}</div>}
        <div className="utility-label">Utilities</div>
        <div className="operation-group">{OPERATIONS.utility.map(([key, title, description, badge]) => <OperationCard key={key} operation={key} title={title} description={description} badge={badge} active={operation === key} onClick={() => { setMode("utility"); setOperation(key); setError(""); }} />)}</div>
      </aside>

      <form className="config-panel" onSubmit={submit}>
        <div className="panel-title"><span>02</span><div><h2>Configuration</h2><p>{operation.replaceAll("_", " ")} inputs</p></div></div>
        {(isEncrypt || operation === "decrypt_veinn" || isPrivate || operation === "derive_veinn") && <div className="form-section">
          <div className="section-title"><span>Required inputs</span><small>* REQUIRED</small></div>
          {(isEncrypt || operation === "decrypt_veinn" || operation === "derive_veinn") && <Field label="Shared seed" name="seed" value={form.seed} onChange={update} hint="The same seed is required to decrypt." />}
          {isEncrypt && <label className="field"><span>Message</span><textarea name="message" value={form.message} onChange={update} placeholder="Write or paste plaintext…" /></label>}
          {operation === "encrypt_public" && <Field label="Recipient public key" name="pubfile" value={form.pubfile} onChange={update} />}
          {operation === "encrypt_public" && <Field label="Input file (optional)" name="in_path" value={form.in_path} onChange={update} />}
          {(operation === "decrypt_veinn" || isPrivate) && <Field label="Encrypted file" name="enc_file" value={form.enc_file} onChange={update} />}
          {isPrivate && <><Field label="Private key file" name="privfile" value={form.privfile} onChange={update} /><Field label="Keystore (optional)" name="keystore" value={form.keystore} onChange={update} /><Field label="Key name (optional)" name="key_name" value={form.key_name} onChange={update} /><Field label="Passphrase (optional)" name="passphrase" type="password" value={form.passphrase} onChange={update} /></>}
        </div>}
        {isKeypair && <div className="form-section"><div className="section-title"><span>Key destination</span><small>* REQUIRED</small></div><Field label="Public key file" name="pubfile" value={form.pubfile} onChange={update} /><label className="choice-row"><input type="checkbox" name="store_private" checked={form.store_private} onChange={update} /><span>Store private key in encrypted keystore</span></label>{form.store_private ? <><Field label="Keystore file" name="keystore" value={form.keystore} onChange={update} /><Field label="Key name" name="key_name" value={form.key_name} onChange={update} /><Field label="Passphrase" name="passphrase" type="password" value={form.passphrase} onChange={update} /></> : <Field label="Private key file" name="privfile" value={form.privfile} onChange={update} />}</div>}
        {isKeystore && <div className="form-section"><div className="section-title"><span>Keystore details</span><small>* REQUIRED</small></div><Field label="Keystore file" name="keystore" value={form.keystore} onChange={update} /><Field label="Passphrase" name="passphrase" type="password" value={form.passphrase} onChange={update} /></div>}
        <button type="button" className="advanced-button" onClick={() => setAdvanced(!advanced)}>{advanced ? "Hide" : "Show"} advanced parameters <span>{advanced ? "−" : "+"}</span></button>
        {advanced && <div className="advanced-grid">
          <RangeField label="Vector size (N)" name="n" value={form.n} onChange={update} min={8} max={2048} step={8} />
          <RangeField label="Rounds" name="rounds" value={form.rounds} onChange={update} min={1} max={64} />
          <RangeField label="Layers / round" name="layers_per_round" value={form.layers_per_round} onChange={update} min={1} max={64} />
          <RangeField label="Shuffle stride" name="shuffle_stride" value={form.shuffle_stride} onChange={update} min={1} max={127} />
          <Field label="Prime modulus (Q)" name="q" type="number" value={form.q} onChange={update} />
          <SelectField label="Chaining" name="mode" value={form.mode} onChange={update}>{["cbc", "ctr", "cfb", "ecb"].map((item) => <option key={item} value={item}>{item.toUpperCase()}</option>)}</SelectField>
          <SelectField label="File format" name="file_type" value={form.file_type} onChange={update}><option value="json">JSON</option><option value="bin">Binary</option></SelectField>
          <label className="choice-row"><input type="checkbox" name="use_lwe" checked={form.use_lwe} onChange={update} /><span>Use LWE PRF</span></label>
        </div>}
        {error && <div className="error-box"><strong>Operation failed</strong><span>{error}</span></div>}
        <button className="primary-button" type="submit" disabled={busy}>{busy ? "Running operation…" : "Run operation"}<span>→</span></button>
      </form>

      <aside className="result-panel">
        <div className="panel-title"><span>03</span><div><h2>Result</h2><p>Output from this session</p></div></div>
        <div className={`result-state ${result ? "has-result" : ""}`}><span className="result-icon">{result ? "✓" : "—"}</span><strong>{result ? "Operation complete" : "Nothing run yet"}</strong><p>{result?.message || "Your output will appear here after running an operation."}</p>{result?.output && <code>{result.output}</code>}</div>
        {result?.plaintext !== undefined && <div className="plaintext"><div className="section-title"><span>Decrypted plaintext</span></div><pre>{result.plaintext}</pre></div>}
        <Log entries={entries} busy={busy} />
      </aside>
    </div>
    <footer><span>VEINN</span> Local research build · no external network access</footer>
  </main>;
}
