    let sessionId = null;
    let sessionData = null;
    let usingDefaultData = false;

    function showToast(message, type = 'info') {
      const container = document.getElementById('toast-container');
      if (!container || typeof bootstrap === 'undefined') {
        alert(message);
        return;
      }
      const toastEl = document.createElement('div');
      let classes = 'toast align-items-center text-white border-0 ';
      if (type === 'success') {
        classes += 'bg-success';
      } else if (type === 'error') {
        classes += 'bg-danger';
      } else if (type === 'warning') {
        classes += 'bg-warning text-dark';
      } else {
        classes += 'bg-info';
      }
      toastEl.className = classes;
      toastEl.role = 'alert';
      toastEl.ariaLive = 'assertive';
      toastEl.ariaAtomic = 'true';
      toastEl.innerHTML = `
        <div class="d-flex">
          <div class="toast-body">${message}</div>
          <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>`;
      container.appendChild(toastEl);
      const t = new bootstrap.Toast(toastEl, { delay: 5000 });
      t.show();
      toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
    }

    window.addEventListener('DOMContentLoaded', async function() {
      try {
        const hasDefault = await checkDefaultData();
        await checkDriverMapping();

        if (hasDefault) {
          useDefaultData();
        } else {
          document.getElementById('config-section').style.display = 'none';
        }

      } catch (error) {
        console.error('Error during page initialization:', error);
      }
    });

    async function checkDefaultData() {
      try {
        const response = await fetch('/check_default_data');
        if (!response.ok) {
          const text = await response.text();
          throw new Error(`Status ${response.status}: ${text}`);
        }
        const data = await response.json();
        return data.has_default;
      } catch (error) {
        console.error('Error checking default data:', error);
        return false;
      }
    }

    async function checkDriverMapping() {
      try {
        const response = await fetch('/check_driver_mapping');
        const data = await response.json();
        const statusDiv = document.getElementById('mapping-status');
        if (statusDiv) {
          if (data.exists) {
            statusDiv.innerHTML = '<div class="success">✅ Driver mapping is available.</div>';
          } else {
            statusDiv.innerHTML = '<div class="warning">⚠️ No driver mapping file found. Upload one to enable Race Pace Estimation.</div>';
          }
        }
        return data.exists;
      } catch (error) {
        console.error('Error checking driver mapping:', error);
        return false;
      }
    }

    function showDriverMapping() {
      const section = document.getElementById('driver-mapping-section');
      if (section) {
        section.style.display = 'block';
      }
    }

    function showUploadSection() {
      const configSection = document.getElementById('config-section');
      if (configSection) configSection.style.display = 'none';
      const uploadSection = document.getElementById('upload-section');
      if (uploadSection) uploadSection.style.display = 'block';
      const resultsSection = document.getElementById('results-section');
      if (resultsSection) resultsSection.style.display = 'none';
      const defaultStatus = document.getElementById('default-data-status');
      if (defaultStatus) defaultStatus.style.display = 'none';

      const driverInput = document.getElementById('driver-data');
      if (driverInput) driverInput.value = '';
      const consInput = document.getElementById('constructor-data');
      if (consInput) consInput.value = '';
      const calInput = document.getElementById('calendar-data');
      if (calInput) calInput.value = '';
      const tracksInput = document.getElementById('tracks-data');
      if (tracksInput) tracksInput.value = '';
      const updateChk = document.getElementById('update-default-checkbox');
      if (updateChk) updateChk.checked = false;
    }

    async function useDefaultData() {
      fetch('/check_default_data')
        .then(async response => {
          if (!response.ok) {
            const text = await response.text();
            throw new Error(`Status ${response.status}: ${text}`);
          }
          return response.json();
        })
        .then(data => {
          if (!data.has_default) return;

          sessionId         = 'default';
          sessionData       = data.data;
          usingDefaultData  = true;

          const racesCompletedEl = document.getElementById('races-completed');
          if (racesCompletedEl) {
            racesCompletedEl.textContent = data.data.races_completed;
          }

          populateTeamSelects(data.data.drivers, data.data.constructors);
          loadConfig();

          document.getElementById('config-section').style.display = 'block';
          const uploadSection = document.getElementById('upload-section');
          if (uploadSection) {
            uploadSection.style.display = 'none';
          }

          checkDriverMapping();
        })
        .catch(error => {
          showToast('Error loading default data: ' + error.message, 'error');
        });
    }

    function populateTeamSelects(drivers, constructors) {
      const driverContainer = document.getElementById('driver-selects');
      if (driverContainer) {
        driverContainer.innerHTML = '';
        for (let i = 0; i < 5; i++) {
          const wrapper = document.createElement('div');
          wrapper.className = 'select-wrapper';
          const select = document.createElement('select');
          select.className = 'driver-select';
          select.innerHTML = '<option value="">Select driver…</option>';
          drivers.forEach(driver => {
            select.innerHTML += `<option value="${driver}">${driver}</option>`;
          });
          select.addEventListener('change', validateUniqueSelections);
          const keep = document.createElement('input');
          keep.type = 'checkbox';
          keep.className = 'driver-keep';
          keep.title = 'Keep';
          wrapper.appendChild(select);
          wrapper.appendChild(keep);
          driverContainer.appendChild(wrapper);
        }
      }

      const constructorContainer = document.getElementById('constructor-selects');
      if (constructorContainer) {
        constructorContainer.innerHTML = '';
        for (let i = 0; i < 2; i++) {
          const wrapper = document.createElement('div');
          wrapper.className = 'select-wrapper';
          const select = document.createElement('select');
          select.className = 'constructor-select';
          select.innerHTML = '<option value="">Select constructor…</option>';
          constructors.forEach(cons => {
            select.innerHTML += `<option value="${cons}">${cons}</option>`;
          });
          select.addEventListener('change', validateUniqueSelections);
          const keep = document.createElement('input');
          keep.type = 'checkbox';
          keep.className = 'constructor-keep';
          keep.title = 'Keep';
          wrapper.appendChild(select);
          wrapper.appendChild(keep);
          constructorContainer.appendChild(wrapper);
        }
      }

      validateUniqueSelections();
      toggleConfigMode();
    }

    function validateUniqueSelections() {
      const drivers = [];
      document.querySelectorAll('.driver-select').forEach(sel => drivers.push(sel.value));
      const driverDuplicates = drivers.filter((val, idx, arr) => val && arr.indexOf(val) !== idx);
      document.querySelectorAll('.driver-select').forEach(sel => {
        if (driverDuplicates.includes(sel.value) && sel.value) {
          sel.classList.add('duplicate-selection');
        } else {
          sel.classList.remove('duplicate-selection');
        }
      });

      const constructors = [];
      document.querySelectorAll('.constructor-select').forEach(sel => constructors.push(sel.value));
      const consDuplicates = constructors.filter((val, idx, arr) => val && arr.indexOf(val) !== idx);
      document.querySelectorAll('.constructor-select').forEach(sel => {
        if (consDuplicates.includes(sel.value) && sel.value) {
          sel.classList.add('duplicate-selection');
        } else {
          sel.classList.remove('duplicate-selection');
        }
      });

      return driverDuplicates.length === 0 && consDuplicates.length === 0;
    }

    function setCookie(name, value, days = 30) {
      const expires = new Date();
      expires.setTime(expires.getTime() + days * 24 * 60 * 60 * 1000);
      document.cookie = `${name}=${JSON.stringify(value)};expires=${expires.toUTCString()};path=/`;
    }

    function getCookie(name) {
      const nameEQ = name + '=';
      const ca = document.cookie.split(';');
      for (let i = 0; i < ca.length; i++) {
        let c = ca[i];
        while (c.charAt(0) === ' ') c = c.substring(1);
        if (c.indexOf(nameEQ) === 0) {
          try {
            return JSON.parse(c.substring(nameEQ.length));
          } catch {
            return null;
          }
        }
      }
      return null;
    }

    function updateSimpleLabel(val) {
      const labels = ['Low Risk','Low-Med','Medium','Med-High','High Risk'];
      document.getElementById('simple-slider-label').textContent = labels[val-1] || '';
    }

    function applySimpleConfig(val) {
      const idx = parseInt(val) - 1;
      if (!simpleMatrix || !simpleMatrix[idx]) return;
      const m = simpleMatrix[idx];
      if (m.pace_weight)        document.getElementById('pace-weight').value = m.pace_weight;
      if (m.pace_modifier_type) document.getElementById('pace-modifier-type').value = m.pace_modifier_type;
      if (m.weighting_scheme)   document.getElementById('weighting-scheme').value = m.weighting_scheme;
      if (m.risk_tolerance)     document.getElementById('risk-tolerance').value = m.risk_tolerance;
      saveConfigToCookie();
    }

    function toggleConfigMode() {
      const isAdvanced = document.getElementById('config-toggle').checked;
      const mode = isAdvanced ? 'advanced' : 'simple';
      document.getElementById('simple-config').style.display = isAdvanced ? 'none' : 'block';
      document.getElementById('advanced-config').style.display = isAdvanced ? 'block' : 'none';
      if (!isAdvanced) applySimpleConfig(document.getElementById('simple-slider').value);
    }

    function saveConfigToCookie() {
      const drivers = [];
      document.querySelectorAll('.driver-select').forEach(sel => {
        if (sel.value) drivers.push(sel.value);
      });
      const keepDrivers = [];
      document.querySelectorAll('.driver-keep').forEach((cb, idx) => {
        if (cb.checked && drivers[idx]) keepDrivers.push(drivers[idx]);
      });
      const constructors = [];
      document.querySelectorAll('.constructor-select').forEach(sel => {
        if (sel.value) constructors.push(sel.value);
      });
      const keepConstructors = [];
      document.querySelectorAll('.constructor-keep').forEach((cb, idx) => {
        if (cb.checked && constructors[idx]) keepConstructors.push(constructors[idx]);
      });
      const config = {
        current_drivers:       drivers,
        current_constructors:  constructors,
        keep_drivers:          keepDrivers,
        keep_constructors:     keepConstructors,
        remaining_budget:      document.getElementById('remaining-budget')?.value,
        step1_swaps:           document.getElementById('step1-swaps')?.value,
        weighting_scheme:      document.getElementById('weighting-scheme')?.value,
        risk_tolerance:        document.getElementById('risk-tolerance')?.value,
        multiplier:            2,
        config_mode:           document.getElementById('config-toggle').checked ? 'advanced' : 'simple',
        simple_slider:         document.getElementById('simple-slider')?.value,
        use_fp2_pace:          true,
        pace_weight:           document.getElementById('pace-weight')?.value,
        pace_modifier_type:    document.getElementById('pace-modifier-type')?.value
      };
      setCookie('f1_optimizer_config', config);
    }

    function loadConfigFromCookie() {
      const config = getCookie('f1_optimizer_config');
      if (config) applyConfig(config);
    }

    function loadConfig() {
      if (serverConfig) {
        applyConfig(serverConfig);
        setCookie('f1_optimizer_config', serverConfig);
      } else {
        loadConfigFromCookie();
      }
    }

    function applyConfig(config) {
      const driverSelects = document.querySelectorAll('.driver-select');
      if (config.current_drivers) {
        config.current_drivers.forEach((driver, idx) => {
          if (driverSelects[idx]) driverSelects[idx].value = driver;
        });
      }
      if (config.keep_drivers) {
        const keepSet = new Set(config.keep_drivers);
        document.querySelectorAll('.driver-keep').forEach((cb, idx) => {
          const val = driverSelects[idx]?.value;
          cb.checked = val && keepSet.has(val);
        });
      }
      const constructorSelects = document.querySelectorAll('.constructor-select');
      if (config.current_constructors) {
        config.current_constructors.forEach((cons, idx) => {
          if (constructorSelects[idx]) constructorSelects[idx].value = cons;
        });
      }
      if (config.keep_constructors) {
        const keepSet = new Set(config.keep_constructors);
        document.querySelectorAll('.constructor-keep').forEach((cb, idx) => {
          const val = constructorSelects[idx]?.value;
          cb.checked = val && keepSet.has(val);
        });
      }
      if (config.remaining_budget) document.getElementById('remaining-budget').value = config.remaining_budget;
      if (config.step1_swaps)       document.getElementById('step1-swaps').value       = config.step1_swaps;
      if (config.weighting_scheme)  document.getElementById('weighting-scheme').value  = config.weighting_scheme;
      if (config.risk_tolerance)    document.getElementById('risk-tolerance').value    = config.risk_tolerance;
        if (config.config_mode) {
          document.getElementById("config-toggle").checked = (config.config_mode === "advanced");
        }
        if (config.simple_slider) {
          document.getElementById("simple-slider").value = config.simple_slider;
          updateSimpleLabel(config.simple_slider);
        }

      if (config.pace_weight)        document.getElementById('pace-weight').value       = config.pace_weight;
      if (config.pace_modifier_type) document.getElementById('pace-modifier-type').value = config.pace_modifier_type;

      validateUniqueSelections();
      toggleConfigMode();
    }

    async function uploadFiles() {
      const files = {
        'calendar.csv': document.getElementById('calendar-data')?.files[0],
        'tracks.csv':   document.getElementById('tracks-data')?.files[0]
      };

      let allFilesSelected = true;
      for (const file of Object.values(files)) {
        if (!file) {
          allFilesSelected = false;
          break;
        }
      }
      if (!allFilesSelected) {
        showToast('Please select both required CSV files', 'error');
        return;
      }

      const formData = new FormData();
      for (const [name, file] of Object.entries(files)) {
        formData.append(name, file, name);
      }
      const updateDefault = document.getElementById('update-default-checkbox')?.checked;
      formData.append('update_default', updateDefault);

      try {
        const response = await fetch('/upload', {
          method: 'POST',
          body:   formData
        });
        const data = await response.json();
        if (!data.success) {
          showToast('Error: ' + data.message, 'error');
          return;
        }

        sessionId   = data.session_id;
        sessionData = data;
        usingDefaultData = false;

        document.getElementById('races-completed').textContent = data.races_completed;
        populateTeamSelects(data.drivers, data.constructors);

        const driverMappingFile = document.getElementById('driver-mapping-file')?.files[0];
        if (driverMappingFile) {
          await uploadDriverMappingFile(driverMappingFile);
        }

        if (data.updated_default) {
          showToast('Files uploaded and saved as default data successfully!', 'success');
          await checkDefaultData();
          await checkDriverMapping();
        } else {
          showToast('Files uploaded successfully!', 'success');
        }

        loadConfig();
        document.getElementById('config-section').style.display = 'block';
        document.getElementById('upload-section').style.display = 'none';
      } catch (error) {
        showToast('Error uploading files: ' + error.message, 'error');
      }
    }

    async function uploadDriverMappingFile(file) {
      const formData = new FormData();
      formData.append('driver_mapping', file);
      try {
        const response = await fetch('/upload_driver_mapping', {
          method: 'POST',
          body:   formData
        });
        const data = await response.json();
        const statusDiv = document.getElementById('mapping-status');
        if (data.success) {
          statusDiv.innerHTML = `<div class="success">✅ ${data.message}</div>`;
        } else {
          statusDiv.innerHTML = `<div class="error">❌ ${data.message}</div>`;
        }
      } catch (error) {
        document.getElementById('mapping-status').innerHTML =
          `<div class="error">❌ Error uploading driver mapping: ${error.message}</div>`;
      }
    }

    async function runOptimization() {
      validateUniqueSelections();
      toggleConfigMode();
      const drivers = [];
      document.querySelectorAll('.driver-select').forEach(sel => {
        if (sel.value) drivers.push(sel.value);
      });
      if (drivers.length !== 5) {
        showToast('Please select exactly 5 drivers', 'error');
        return;
      }
      if (new Set(drivers).size !== drivers.length) {
        showToast('Each driver must be unique', 'error');
        return;
      }

      const constructors = [];
      document.querySelectorAll('.constructor-select').forEach(sel => {
        if (sel.value) constructors.push(sel.value);
      });
      if (constructors.length !== 2) {
        showToast('Please select exactly 2 constructors', 'error');
        return;
      }
      if (new Set(constructors).size !== constructors.length) {
        showToast('Each constructor must be unique', 'error');
        return;
      }

      const useFP2 = true;
      const mappingCheck = await fetch('/check_driver_mapping');
      const mappingData  = await mappingCheck.json();
      if (!mappingData.exists) {
        showToast('Driver mapping file is required for Race Pace Estimation. Please upload it first.', 'error');
        return;
      }

      const keepDrivers = [];
      document.querySelectorAll('.driver-keep').forEach((cb, idx) => {
        if (cb.checked && drivers[idx]) keepDrivers.push(drivers[idx]);
      });
      const keepConstructors = [];
      document.querySelectorAll('.constructor-keep').forEach((cb, idx) => {
        if (cb.checked && constructors[idx]) keepConstructors.push(constructors[idx]);
      });

      const config = {
        session_id:          sessionId,
        current_drivers:     drivers,
        current_constructors: constructors,
        keep_drivers:        keepDrivers,
        keep_constructors:   keepConstructors,
        remaining_budget:    document.getElementById('remaining-budget')?.value,
        step1_swaps:         document.getElementById('step1-swaps')?.value,
        weighting_scheme:    document.getElementById('weighting-scheme')?.value,
        risk_tolerance:      document.getElementById('risk-tolerance')?.value,
        multiplier:          2,
        use_fp2_pace:        useFP2,
        pace_weight:         parseFloat(document.getElementById('pace-weight')?.value),
        pace_modifier_type:  document.getElementById('pace-modifier-type')?.value
      };

      saveConfigToCookie();

      const progressSection = document.getElementById('progress-section');
      progressSection.style.display = 'block';
      document.getElementById('results-section').style.display = 'none';
      setTimeout(() => {
        progressSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 50);

      try {
        const response = await fetch('/optimize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config)
        });
        const data = await response.json();
        if (data.success) {
          displayResults(data);
        } else {
          showToast('Error: ' + data.message, 'error');
        }
      } catch (error) {
        showToast('Error running optimization: ' + error.message, 'error');
      } finally {
        document.getElementById('progress-section').style.display = 'none';
      }
    }

    async function scheduleOptimization() {
      validateUniqueSelections();
      toggleConfigMode();
      const drivers = [];
      document.querySelectorAll('.driver-select').forEach(sel => { if (sel.value) drivers.push(sel.value); });
      if (drivers.length !== 5) { showToast('Please select exactly 5 drivers', 'error'); return; }
      if (new Set(drivers).size !== drivers.length) { showToast('Each driver must be unique', 'error'); return; }
      const constructors = [];
      document.querySelectorAll('.constructor-select').forEach(sel => { if (sel.value) constructors.push(sel.value); });
      if (constructors.length !== 2) { showToast('Please select exactly 2 constructors', 'error'); return; }
      if (new Set(constructors).size !== constructors.length) { showToast('Each constructor must be unique', 'error'); return; }
      const useFP2 = true;
      const keepDrivers = [];
      document.querySelectorAll('.driver-keep').forEach((cb, idx) => { if (cb.checked && drivers[idx]) keepDrivers.push(drivers[idx]); });
      const keepConstructors = [];
      document.querySelectorAll('.constructor-keep').forEach((cb, idx) => { if (cb.checked && constructors[idx]) keepConstructors.push(constructors[idx]); });
      const config = {
        session_id: sessionId,
        current_drivers: drivers,
        current_constructors: constructors,
        keep_drivers: keepDrivers,
        keep_constructors: keepConstructors,
        remaining_budget: document.getElementById('remaining-budget')?.value,
        step1_swaps: document.getElementById('step1-swaps')?.value,
        weighting_scheme: document.getElementById('weighting-scheme')?.value,
        risk_tolerance: document.getElementById('risk-tolerance')?.value,
        multiplier: 2,
        use_fp2_pace: useFP2,
        pace_weight: parseFloat(document.getElementById('pace-weight')?.value),
        pace_modifier_type: document.getElementById('pace-modifier-type')?.value
      };
      saveConfigToCookie();
      try {
        if (autoEmailOptIn) {
          const update = confirm('You are currently opted in for automatic email optimisations.\nOK to update defaults, Cancel for one-off.');
          if (update) {
            const r = await fetch('/auto_email_config', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ opt_in: true, config })
            });
            if ((await r.json()).success) {
              showToast('Default optimisation settings updated.', 'success');
            }
            return;
          }
        } else {
          const opt = confirm('Opt-in to receive optimisations for every race?');
          if (opt) {
            const r = await fetch('/auto_email_config', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ opt_in: true, config })
            });
            if ((await r.json()).success) {
              showToast('Automatic email optimisation enabled.', 'success');
              autoEmailOptIn = true;
            }
            return;
          }
        }

        const resp = await fetch('/schedule_optimization', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config)
        });
        const data = await resp.json();
        if (data.success) {
          showToast('Optimisation scheduled. Results will be emailed to you.', 'success');
        } else {
          showToast('Error: ' + data.message, 'error');
        }
      } catch (err) {
        showToast('Error scheduling optimisation: ' + err.message, 'error');
      }
    }

    function displayResults(data) {
      const resultsContent = document.getElementById('results-content');
      const opt = data.optimization;
      let fp2InfoHtml = '';

      // Only show FP2 block if “applied” is true
      if (opt.fp2_info && opt.fp2_info.applied) {
        fp2InfoHtml = `
          <div class="result-card" style="border-left-color: #17a2b8;">
            <h3>Race Pace Estimation</h3>
            <p><strong>Pace Weight:</strong> ${(opt.fp2_info.pace_weight * 100).toFixed(0)}% current form, ${(100 - opt.fp2_info.pace_weight * 100).toFixed(0)}% historical</p>
            <p><strong>Modifier Type:</strong> ${opt.fp2_info.modifier_type}</p>
            <p style="color: #28a745; font-weight: bold;">✅ FP2 Lap data successfully integrated</p>
            ${opt.fp2_info.pace_adjustments && opt.fp2_info.pace_adjustments.length > 0 ? `
              <h4>Pace Adjustments for Current Team:</h4>
              ${opt.fp2_info.pace_adjustments.map(adj => `
                <div class="pace-adjustment">
                  <span>${adj.driver}</span>
                  <span class="pace-score">
                    Score: ${adj.pace_score} | 
                    Modifier: ${adj.pace_modifier}× | 
                    VFM: ${adj.vfm_original} → ${adj.vfm_adjusted}
                  </span>
                </div>
              `).join('')}
            ` : ''}
          </div>
        `;
      }

      const html = `
        ${fp2InfoHtml}
        <div class="result-card">
          <h3>Next Race: ${opt.step1.circuit}</h3>
          ${opt.step1.swaps.length > 0 ? `
            <h4>Recommended Swaps:</h4>
            ${opt.step1.swaps.map(swap => `
              <div class="swap-item">
                <span>${swap[0]}: ${swap[1]} → ${swap[2]}</span>
              </div>
            `).join('')}
          ` : '<p>No swaps recommended</p>'}
          <p><strong>Expected Points:</strong> ${opt.step1.expected_points.toFixed(1)} 
             <span class="improvement">(+${opt.step1.improvement.toFixed(1)})</span></p>
          <p><strong>Boost Driver:</strong> ${opt.step1.boost_driver}</p>
        </div>

        <div class="result-card">
          <h3>Following Race: ${opt.step2.circuit}</h3>
          ${opt.step2.swaps.length > 0 ? `
            <h4>Additional Swaps:</h4>
            ${opt.step2.swaps.map(swap => `
              <div class="swap-item">
                <span>${swap[0]}: ${swap[1]} → ${swap[2]}</span>
              </div>
            `).join('')}
          ` : '<p>No additional swaps</p>'}
          <p><strong>Expected Points:</strong> ${opt.step2.expected_points.toFixed(1)} 
             <span class="improvement">(+${opt.step2.improvement.toFixed(1)})</span></p>
          <p><strong>Boost Driver:</strong> ${opt.step2.boost_driver}</p>
          <p><strong>Budget:</strong> ${opt.step2.budget_used.toFixed(1)}M used,
             ${opt.step2.budget_remaining.toFixed(1)}M remaining</p>
        </div>

        <div class="result-card">
          <h3>Final Team</h3>
          <h4>Drivers:</h4>
          <div class="team-display">
            ${opt.step2.team.drivers.map(d => `<div class="team-member">${d}</div>`).join('')}
          </div>
          <h4>Constructors:</h4>
          <div class="team-display">
            ${opt.step2.team.constructors.map(c => `<div class="team-member">${c}</div>`).join('')}
          </div>
        </div>

        <div class="result-card">
          <h3>Summary</h3>
          <p><strong>Total Improvement:</strong> <span class="improvement">+${opt.summary.total_improvement.toFixed(1)} points</span></p>
          <p><strong>Patterns Evaluated:</strong> ${opt.summary.patterns_evaluated.toLocaleString()}</p>
          <p><strong>Optimization Time:</strong> ${opt.summary.optimization_time.toFixed(1)}s</p>
          ${opt.fp2_info && opt.fp2_info.applied ?
            '<p style="color: #17a2b8;"><strong>Enhancement:</strong> FP2-derived race pace estimations</p>' :
            '<p style="color: #6c757d;"><strong>Analysis:</strong> Historical data only</p>'
          }
        </div>
      `;
      resultsContent.innerHTML = html;
  document.getElementById('results-section').style.display = 'block';
  }
    function toggleNav() {
      document.querySelector('.nav-links').classList.toggle('open');
    }
