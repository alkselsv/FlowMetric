(() => {
  const $ = selector => document.querySelector(selector);
  const state = {
    data: null, time: 0, selectedEvent: null, revealedEvents: 0,
    scrubbing: false, pendingSeek: null, analyticsTab: 'visitors'
  };
  const eventNames = {
    leave_kitchen: 'Выход из кухни',
    seated: 'Посадка за стол', service_start: 'Обслуживание началось', service_end: 'Обслуживание завершено'
  };

  function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
  function formatTime(value) {
    const minutes = Math.floor(value / 60);
    const seconds = value - minutes * 60;
    return `${String(minutes).padStart(2, '0')}:${seconds.toFixed(1).padStart(4, '0')}`;
  }
  function formatDuration(value) {
    if (!Number.isFinite(value) || value < 0) return '—';
    const seconds = Math.floor(value);
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
  }
  function trackRole(person) {
    return state.data.tracks.find(item => String(item.id) === String(person))?.role || '';
  }
  function metricTime(value) {
    if (!Number.isFinite(value) || value < 0) return '—';
    if (value < 60) return `${value.toFixed(1)} с`;
    return `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
  }
  function metricPercent(value) {
    return Number.isFinite(value) ? `${Math.round(value * 100)}%` : '—';
  }
  function pointAt(track, time) {
    const points = track.points;
    if (!points.length || time < points[0][0] || time > points[points.length - 1][0]) return null;
    let low = 0, high = points.length - 1;
    while (low <= high) { const mid = (low + high) >> 1; if (points[mid][0] <= time) low = mid + 1; else high = mid - 1; }
    const a = points[Math.max(0, high)], b = points[Math.min(points.length - 1, high + 1)];
    if (!b || b[0] === a[0] || b[0] - a[0] > 0.7) return Math.abs(time - a[0]) <= 0.25 ? a : null;
    const q = (time - a[0]) / (b[0] - a[0]);
    return [time, a[1] + (b[1] - a[1]) * q, a[2] + (b[2] - a[2]) * q];
  }
  function eventSubject(event) {
    return event.staff != null ? `ID ${event.staff} → ID ${event.customer}` : `ID ${event.person}`;
  }
  function eventTitle(event) {
    if (event.type === 'enter' || event.type === 'exit') {
      const track = state.data.tracks.find(item => String(item.id) === String(event.person));
      const subject = track?.role === 'staff' ? 'сотрудника' : track?.role === 'customer' ? 'посетителя' : 'человека';
      return `${event.type === 'enter' ? 'Вход' : 'Выход'} ${subject}`;
    }
    return eventNames[event.type] || event.type;
  }
  function eventClass(type) { return type.startsWith('service_') ? 'service' : ['enter', 'exit', 'leave_kitchen'].includes(type) ? 'movement' : 'state'; }

  function renderEvents() {
    const feed = $('#event-feed'); feed.replaceChildren();
    state.data.events.forEach((event, index) => {
      const button = document.createElement('button');
      button.type = 'button'; button.className = `event ${eventClass(event.type)}`; button.dataset.index = index; button.hidden = true;
      button.innerHTML = `<span class="event-time">${formatTime(event.t).slice(3)}</span><i class="event-dot"></i><span><span class="event-title">${eventTitle(event)}</span><span class="event-subtitle">${eventSubject(event)}</span></span>`;
      button.addEventListener('click', () => { state.selectedEvent = index; setTime(Number(event.t)); });
      feed.append(button);
    });
    const marks = $('#timeline-events');
    state.data.events.forEach((event, index) => {
      const mark = document.createElement('i'); mark.className = 'timeline-event'; mark.dataset.index = index;
      mark.style.left = `${event.t / state.data.meta.duration * 100}%`; mark.hidden = true; marks.append(mark);
    });
  }
  function eventWait(event) {
    const person = event.customer ?? event.person;
    if (person == null || trackRole(person) !== 'customer') return null;
    const eventTime = Number(event.t);
    const waiting = state.data.states.find(item => item.type === 'waiting' && String(item.person) === String(person) && Number(item.start) <= eventTime && eventTime <= Number(item.end));
    return waiting ? eventTime - Number(waiting.start) : null;
  }
  function waitingMetrics(time) {
    const people = new Set();
    let maximum = null;
    state.data.states.forEach(item => {
      if (item.type !== 'waiting' || trackRole(item.person) !== 'customer' || Number(item.start) > time || time > Number(item.end)) return;
      people.add(String(item.person));
      maximum = Math.max(maximum ?? 0, time - Number(item.start));
    });
    return { count: people.size, maximum };
  }
  function updateDetail(time) {
    let selected = null;
    if (state.selectedEvent != null && Number(state.data.events[state.selectedEvent]?.t) <= time) {
      selected = state.data.events[state.selectedEvent];
    } else {
      const revealed = state.data.events.filter(event => Number(event.t) <= time);
      selected = revealed.at(-1) || null;
    }
    const maxWait = waitingMetrics(time).maximum;
    const slaSeconds = Number(state.data.meta.waiting_sla_seconds ?? 180);
    const slaLabel = formatDuration(slaSeconds);
    const badge = $('#sla-badge');
    badge.className = `sla-badge ${maxWait == null ? 'idle' : maxWait >= slaSeconds ? 'breach' : 'ok'}`;
    badge.textContent = maxWait == null ? `SLA ${slaLabel}: нет ожидания` : maxWait >= slaSeconds ? `SLA ${slaLabel}: нарушен` : `SLA ${slaLabel}: в норме`;
    if (!selected) {
      $('#detail-title').textContent = 'Ожидание первого события';
      $('#detail-subject').textContent = 'События появятся во время воспроизведения';
      $('#detail-time').textContent = '—'; $('#detail-table').textContent = '—';
      $('#detail-wait').textContent = '—'; $('#detail-confidence').textContent = '—';
      return;
    }
    const confidence = Number(selected.confidence ?? selected.verifier_confidence);
    $('#detail-title').textContent = eventTitle(selected);
    $('#detail-subject').textContent = eventSubject(selected);
    $('#detail-time').textContent = formatTime(Number(selected.t));
    $('#detail-table').textContent = selected.table != null ? `№ ${selected.table}` : '—';
    $('#detail-wait').textContent = metricTime(eventWait(selected));
    $('#detail-confidence').textContent = Number.isFinite(confidence) ? metricPercent(confidence) : '—';
  }
  function observedVisits(events) {
    const visits = [];
    events.forEach(event => {
      const person = String(event.person ?? '');
      if (trackRole(person) !== 'customer') return;
      if (event.type === 'enter') {
        visits.push({ person, entered: Number(event.t), seated: null, exited: null });
        return;
      }
      if (!['seated', 'exit'].includes(event.type)) return;
      const visit = [...visits].reverse().find(item => item.person === person && item.exited == null && (event.type !== 'seated' || item.seated == null));
      if (!visit) return;
      if (event.type === 'seated') visit.seated = Number(event.t);
      else visit.exited = Number(event.t);
    });
    return visits;
  }
  function visitorMetrics(time) {
    const events = state.data.events.filter(event => Number(event.t) <= time);
    const visits = observedVisits(events);
    const exited = events.filter(event => event.type === 'exit' && trackRole(event.person) === 'customer');
    const seatedVisits = visits.filter(visit => visit.seated != null);
    const seatTimes = seatedVisits.map(visit => visit.seated - visit.entered);
    return {
      metrics: [
        ['Вошли', visits.length],
        ['Вышли', exited.length],
        ['Конверсия в посадку', visits.length ? metricPercent(seatedVisits.length / visits.length) : '—'],
        ['Среднее до посадки', seatTimes.length ? metricTime(seatTimes.reduce((sum, value) => sum + value, 0) / seatTimes.length) : '—'],
      ],
      note: 'Каждый подтверждённый вход считается отдельным визитом; посетители, уже находившиеся в зале в начале записи, не включены.',
    };
  }
  function serviceMetrics(time) {
    const starts = state.data.events.filter(event => event.type === 'service_start' && Number(event.t) <= time);
    const completed = state.data.service_intervals.filter(item => {
      const rawStart = item.start ?? item.start_time ?? item.t0;
      const rawEnd = item.end ?? item.end_time ?? item.t1;
      if (rawStart == null || rawEnd == null) return false;
      const start = Number(rawStart);
      const end = Number(rawEnd);
      return Number.isFinite(start) && Number.isFinite(end) && end <= time;
    });
    const durations = completed.map(item => Number(item.end ?? item.end_time ?? item.t1) - Number(item.start ?? item.start_time ?? item.t0));
    const contacts = new Map();
    starts.filter(event => event.table != null).forEach(event => contacts.set(String(event.table), (contacts.get(String(event.table)) || 0) + 1));
    const repeats = [...contacts.values()].reduce((sum, count) => sum + Math.max(0, count - 1), 0);
    return {
      metrics: [
        ['Подходы', starts.length],
        ['Завершено', completed.length],
        ['Средний контакт', durations.length ? metricTime(durations.reduce((sum, value) => sum + value, 0) / durations.length) : '—'],
        ['Повторные контакты', repeats],
      ],
      note: 'Длительность — только по наблюдаемым границам; повторные контакты группируются по устойчивому столу.',
    };
  }
  function tableMetrics(time) {
    const seatings = state.data.events.filter(event => event.type === 'seated' && Number(event.t) <= time && trackRole(event.person) === 'customer').length;
    const occupancyAvailable = Array.isArray(state.data.table_occupancy);
    const occupancy = occupancyAvailable ? state.data.table_occupancy : [];
    const occupied = new Set(occupancy.filter(item => {
      const start = Number(item.start), end = Number(item.end);
      return start <= time && (time < end || (item.right_censored && time <= end));
    }).map(item => String(item.table))).size;
    const turnover = occupancy.filter(item => !item.right_censored && Number(item.end) <= time).length;
    const grace = formatDuration(Number(state.data.meta.table_vacancy_grace_seconds ?? 0));
    return {
      metrics: [
        ['Столов на карте', state.data.tables.length || '—'],
        ['Посадок', seatings],
        ['Занято сейчас', occupancyAvailable ? occupied : '—'],
        ['Оборачиваемость', occupancyAvailable ? turnover : '—'],
      ],
      note: occupancyAvailable
        ? `Стол освобождается после ухода всех закреплённых посетителей и защитного окна ${grace}. Оборачиваемость — число завершённых циклов занятости.`
        : 'Перезапустите viewer, чтобы сервер сформировал интервалы занятости столов.',
    };
  }
  function qualityMetrics() {
    const quality = state.data.quality || {};
    return {
      metrics: [
        ['Непрерывность ReID', metricPercent(quality.identity_continuity)],
        ['Точность ролей', metricPercent(quality.role_accuracy)],
        ['F1 событий', metricPercent(quality.event_f1)],
        ['tIoU ожидания', metricPercent(quality.waiting_tiou)],
      ],
      note: Object.keys(quality).length ? 'Общие оценки по всем размеченным людям из benchmark_report.json текущего запуска.' : 'Для текущего запуска benchmark_report.json не найден.',
    };
  }
  function renderAnalytics(time) {
    const sections = {
      visitors: () => visitorMetrics(time),
      service: () => serviceMetrics(time),
      tables: () => tableMetrics(time),
      quality: qualityMetrics,
    };
    const section = sections[state.analyticsTab]();
    const metrics = $('#analytics-metrics');
    metrics.replaceChildren(...section.metrics.map(([label, value]) => {
      const item = document.createElement('div');
      const strong = document.createElement('strong'); strong.textContent = value;
      const span = document.createElement('span'); span.textContent = label;
      item.append(strong, span); return item;
    }));
    $('#analytics-note').textContent = section.note;
    $('#analytics-time').textContent = `на ${formatTime(time)}`;
  }
  function renderDynamic() {
    const time = state.time;
    let visible = 0, customers = 0, staff = 0;
    state.data.tracks.forEach(track => {
      const point = pointAt(track, time); if (!point) return;
      visible += 1;
      if (track.role === 'customer') customers += 1;
      if (track.role === 'staff') staff += 1;
    });
    const waiting = waitingMetrics(time);
    const servicesStarted = state.data.events.filter(event => event.type === 'service_start' && Number(event.t) <= time).length;
    const revealed = state.data.events.filter(event => event.t <= time).length;
    $('#visible-count').textContent = visible; $('#customer-count').textContent = customers;
    $('#staff-count').textContent = staff; $('#waiting-count').textContent = waiting.count;
    $('#service-count').textContent = servicesStarted; $('#max-wait').textContent = formatDuration(waiting.maximum);
    $('#event-count').textContent = revealed;
    document.querySelectorAll('.event').forEach((element, index) => {
      const isRevealed = state.data.events[index].t <= time;
      element.hidden = !isRevealed;
      element.classList.toggle('past', isRevealed);
      element.classList.toggle('selected', index === state.selectedEvent);
    });
    document.querySelectorAll('.timeline-event').forEach((element, index) => {
      element.hidden = state.data.events[index].t > time;
    });
    if (revealed > state.revealedEvents && !$('#video').paused) {
      const latest = document.querySelector(`.event[data-index="${revealed - 1}"]`);
      latest?.scrollIntoView({ block: 'nearest' });
    }
    state.revealedEvents = revealed;
    $('#clock').textContent = `${formatTime(time)} / ${formatTime(state.data.meta.duration)}`;
    $('#timeline').value = time; updateDetail(time); renderAnalytics(time);
  }
  function setTime(value, seekVideo = true) {
    state.time = clamp(value, 0, state.data.meta.duration);
    const video = $('#video');
    if (seekVideo) {
      state.pendingSeek = state.time;
      if (video.readyState > 0 && Math.abs(video.currentTime - state.time) > 0.04) {
        video.currentTime = state.time;
      }
    }
    renderDynamic();
  }
  function bind() {
    const video = $('#video'), play = $('#play');
    const timeline = $('#timeline');
    const markVideoReady = () => { $('#video-loading').hidden = true; };
    const beginScrub = () => { state.scrubbing = true; };
    const finishScrub = () => {
      if (!state.scrubbing) return;
      state.scrubbing = false;
      setTime(Number(timeline.value));
    };
    timeline.addEventListener('pointerdown', beginScrub);
    timeline.addEventListener('touchstart', beginScrub, { passive: true });
    timeline.addEventListener('input', event => {
      state.selectedEvent = null;
      setTime(Number(event.target.value));
    });
    timeline.addEventListener('change', finishScrub);
    timeline.addEventListener('pointerup', finishScrub);
    timeline.addEventListener('pointercancel', finishScrub);
    play.addEventListener('click', async () => {
      if (video.paused) { if (video.currentTime >= state.data.meta.duration) video.currentTime = 0; await video.play(); }
      else video.pause();
    });
    video.addEventListener('play', () => { play.textContent = 'Ⅱ'; play.setAttribute('aria-label', 'Пауза'); });
    video.addEventListener('play', () => { $('#playback-status').textContent = 'воспроизведение'; });
    video.addEventListener('pause', () => {
      play.textContent = '▶'; play.setAttribute('aria-label', 'Воспроизвести');
      $('#playback-status').textContent = video.ended ? 'завершено' : 'пауза';
    });
    video.addEventListener('ended', () => { $('#playback-status').textContent = 'завершено'; });
    video.addEventListener('timeupdate', () => {
      if (state.scrubbing) return;
      if (state.pendingSeek != null && Math.abs(video.currentTime - state.pendingSeek) > 0.2) return;
      state.pendingSeek = null;
      state.time = video.currentTime;
      renderDynamic();
    });
    video.addEventListener('seeked', () => {
      state.pendingSeek = null;
      state.time = video.currentTime;
      renderDynamic();
    });
    video.addEventListener('loadedmetadata', () => {
      markVideoReady();
      video.currentTime = (state.pendingSeek ?? state.time) || Math.min(0.01, video.duration || 0);
    });
    video.addEventListener('loadeddata', markVideoReady);
    video.addEventListener('canplay', markVideoReady);
    video.addEventListener('error', () => { $('#video-loading').hidden = false; $('#video-loading').textContent = 'Видео не удалось загрузить'; });
    $('#speed').addEventListener('change', event => { video.playbackRate = Number(event.target.value); });
    document.querySelectorAll('.analytics-tab').forEach(tab => tab.addEventListener('click', () => {
      state.analyticsTab = tab.dataset.analyticsTab;
      document.querySelectorAll('.analytics-tab').forEach(item => {
        const active = item === tab;
        item.classList.toggle('active', active);
        item.setAttribute('aria-selected', String(active));
      });
      renderAnalytics(state.time);
    }));
  }
  async function init() {
    try {
      const response = await fetch('/api/session'); if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.data = await response.json();
      $('#mode').textContent = `${state.data.meta.run} · ${state.data.meta.mode || 'local'}`;
      const video = $('#video');
      $('#timeline').max = state.data.meta.duration; renderEvents(); bind(); renderDynamic();
      if (state.data.meta.video?.available) { video.src = state.data.meta.video.url; $('#projection-label').textContent = state.data.meta.video.name; }
      else { $('#video-loading').textContent = 'Размеченное видео не найдено'; $('#projection-label').textContent = 'Укажите файл через --video'; }
    } catch (error) { const box = $('#error'); box.hidden = false; box.textContent = `Не удалось загрузить данные: ${error.message}`; }
  }
  init();
})();
