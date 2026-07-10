const STORAGE_KEY = "microscope-sharing-platform:v1";
const ADMIN_PASSCODE = "micro2026";
const ADMIN_EMAIL = "microscope@example.com";
const REMOTE_EVENT_ENDPOINT = "";

const emptyStore = () => ({
  visits: 0,
  firstVisitAt: null,
  lastVisitAt: null,
  events: [],
  bookings: []
});

function readStore() {
  try {
    return { ...emptyStore(), ...JSON.parse(localStorage.getItem(STORAGE_KEY)) };
  } catch {
    return emptyStore();
  }
}

function writeStore(store) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
}

function timestamp() {
  return new Date().toISOString();
}

function trackEvent(type, label, meta = {}) {
  const store = readStore();
  const event = {
    id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`,
    type,
    label,
    meta,
    path: location.pathname,
    at: timestamp()
  };
  store.events.unshift(event);
  store.events = store.events.slice(0, 500);
  writeStore(store);
  sendRemoteEvent(event);
  refreshAnalytics();
}

function sendRemoteEvent(event) {
  if (!REMOTE_EVENT_ENDPOINT) return;
  fetch(REMOTE_EVENT_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(event),
    keepalive: true
  }).catch(() => {});
}

function trackVisit() {
  const store = readStore();
  store.visits += 1;
  store.firstVisitAt ||= timestamp();
  store.lastVisitAt = timestamp();
  writeStore(store);
  trackEvent("view", "page_view", {
    title: document.title,
    referrer: document.referrer || "direct"
  });
}

function wireClickTracking() {
  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-track]");
    if (!target) return;
    trackEvent("click", target.dataset.track, {
      text: target.textContent.trim().slice(0, 80)
    });
  });
}

function serializeForm(form) {
  return Object.fromEntries(new FormData(form).entries());
}

function createBookingId() {
  const date = new Date();
  const stamp = date
    .toISOString()
    .replace(/[-:TZ.]/g, "")
    .slice(0, 14);
  return `MS-${stamp}`;
}

function bookingMailto(booking) {
  const subject = encodeURIComponent(`超景深显微镜预约申请 ${booking.id}`);
  const body = encodeURIComponent(
    [
      `预约编号：${booking.id}`,
      `姓名：${booking.name}`,
      `科室/课题组：${booking.department}`,
      `联系方式：${booking.contact}`,
      `使用日期：${booking.date}`,
      `开始时间：${booking.startTime}`,
      `预计时长：${booking.duration}`,
      `样品类型：${booking.sampleType || "未填写"}`,
      `使用目的：${booking.purpose}`,
      `备注：${booking.notes || "无"}`,
      `提交时间：${new Date(booking.createdAt).toLocaleString("zh-CN")}`
    ].join("\n")
  );
  return `mailto:${ADMIN_EMAIL}?subject=${subject}&body=${body}`;
}

function wireBookingForm() {
  const form = document.querySelector("#bookingForm");
  const status = document.querySelector("#formStatus");
  if (!form || !status) return;

  const dateInput = form.elements.date;
  if (dateInput) dateInput.min = new Date().toISOString().slice(0, 10);

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;

    const booking = {
      id: createBookingId(),
      ...serializeForm(form),
      createdAt: timestamp()
    };
    const store = readStore();
    store.bookings.unshift(booking);
    store.bookings = store.bookings.slice(0, 200);
    writeStore(store);

    trackEvent("booking", "booking_submit", {
      id: booking.id,
      date: booking.date,
      purpose: booking.purpose
    });

    status.textContent = `预约已记录：${booking.id}`;
    form.reset();
    refreshAnalytics();
    window.location.href = bookingMailto(booking);
  });
}

function wireScheduleSlots() {
  const form = document.querySelector("#bookingForm");
  const status = document.querySelector("#formStatus");
  if (!form) return;

  document.querySelectorAll("[data-booking-slot]").forEach((slot) => {
    slot.addEventListener("click", () => {
      const { date, time } = slot.dataset;
      if (date) form.elements.date.value = date;
      if (time) form.elements.startTime.value = time;

      status.textContent = `已选择 ${date} ${time} 的可预约时段，请补全预约信息。`;
      trackEvent("click", "schedule_slot_selected", { date, time });
      form.elements.name.focus();
    });
  });
}

function unlockAnalytics() {
  const input = document.querySelector("#analyticsPasscode");
  const panel = document.querySelector("#analyticsPanel");
  const lock = document.querySelector("#analyticsLock");
  if (!input || !panel || !lock) return;

  if (input.value.trim() !== ADMIN_PASSCODE) {
    input.setCustomValidity("口令不正确");
    input.reportValidity();
    input.setCustomValidity("");
    trackEvent("admin", "analytics_unlock_failed");
    return;
  }

  panel.hidden = false;
  lock.hidden = true;
  trackEvent("admin", "analytics_unlocked");
  refreshAnalytics();
}

function refreshAnalytics() {
  const panel = document.querySelector("#analyticsPanel");
  if (!panel || panel.hidden) return;

  const store = readStore();
  const clicks = store.events.filter((event) => event.type === "click").length;
  const bookings = store.bookings.length;
  const rate = store.visits ? Math.round((bookings / store.visits) * 100) : 0;

  document.querySelector("#metricViews").textContent = String(store.visits);
  document.querySelector("#metricClicks").textContent = String(clicks);
  document.querySelector("#metricBookings").textContent = String(bookings);
  document.querySelector("#metricRate").textContent = `${rate}%`;

  const rows = store.events.slice(0, 12).map((event) => {
    const time = new Date(event.at).toLocaleString("zh-CN");
    return `<tr><td>${escapeHtml(time)}</td><td>${escapeHtml(event.type)}</td><td>${escapeHtml(event.label)}</td></tr>`;
  });
  document.querySelector("#eventsTable").innerHTML =
    rows.join("") || '<tr><td colspan="3">暂无记录</td></tr>';
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    const map = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;"
    };
    return map[char];
  });
}

function toCsv(rows) {
  return rows
    .map((row) =>
      row
        .map((value) => `"${String(value ?? "").replaceAll('"', '""')}"`)
        .join(",")
    )
    .join("\n");
}

function downloadCsv(filename, csv) {
  const blob = new Blob([`\uFEFF${csv}`], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function exportStats() {
  const store = readStore();
  const rows = [["id", "time", "type", "label", "path", "meta"]];
  store.events.forEach((event) => {
    rows.push([event.id, event.at, event.type, event.label, event.path, JSON.stringify(event.meta)]);
  });
  downloadCsv("microscope-analytics.csv", toCsv(rows));
}

function exportBookings() {
  const store = readStore();
  const rows = [
    [
      "id",
      "createdAt",
      "name",
      "department",
      "contact",
      "date",
      "startTime",
      "duration",
      "sampleType",
      "purpose",
      "notes"
    ]
  ];
  store.bookings.forEach((booking) => {
    rows.push([
      booking.id,
      booking.createdAt,
      booking.name,
      booking.department,
      booking.contact,
      booking.date,
      booking.startTime,
      booking.duration,
      booking.sampleType,
      booking.purpose,
      booking.notes
    ]);
  });
  downloadCsv("microscope-bookings.csv", toCsv(rows));
}

function clearLocalData() {
  if (!confirm("确认清空本浏览器保存的统计和预约记录？")) return;
  localStorage.removeItem(STORAGE_KEY);
  trackVisit();
  refreshAnalytics();
}

function wireAnalytics() {
  document.querySelector("#unlockAnalytics")?.addEventListener("click", unlockAnalytics);
  document.querySelector("#analyticsPasscode")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") unlockAnalytics();
  });
  document.querySelector("#exportStats")?.addEventListener("click", exportStats);
  document.querySelector("#exportBookings")?.addEventListener("click", exportBookings);
  document.querySelector("#clearLocalData")?.addEventListener("click", clearLocalData);
}

document.addEventListener("DOMContentLoaded", () => {
  wireClickTracking();
  wireBookingForm();
  wireScheduleSlots();
  wireAnalytics();
  trackVisit();
});
