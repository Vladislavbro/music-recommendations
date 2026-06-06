"use strict";

// 1. Конфиги
const CONFIG = {
  viewBox: { x: 0, y: 0, width: 1920, height: 1080 },
  userIdsUrl: "data/yambda_50m_unique_user_ids.txt",
  mapUrl: "assets/club_map.svg",
  backendUrl: "http://127.0.0.1:8001/recommend",
  recommendationMethod: "audio_agree",
  recommendationTopN: 10,
  clubCapacity: 300,
  maxVerandaShare: 0.2,
  maxToiletShare: 0.05,
  initialQueueSize: 0,
  targetQueueSize: 120,
  visibleQueueLimit: 120,
  queueSpawnIntervalMs: 380,
  admissionIntervalMs: 1050,
  recommendationsIntervalMs: 4000,
  bleUpdateIntervalMs: 450,
  uiUpdateIntervalMs: 300,
  routeArrivalDistance: 5,
  routeChaosMinSegment: 95,
  routeChaosOffset: 26,
  agentRadius: 8,
  selectedAgentRadius: 12,
  defaultSpeedMultiplier: 1,
  speedMin: 0.25,
  speedMax: 5
};

const CLUB_CAPACITY = CONFIG.clubCapacity;
const MAX_VERANDA_SHARE = CONFIG.maxVerandaShare;
const MAX_TOILET_SHARE = CONFIG.maxToiletShare;

const BLE_CONFIG = {
  txPower: -42,
  referenceDistance: 30,
  pathLossExponent: 2.0,
  noiseMin: -5,
  noiseMax: 5,
  minRssi: -95,
  maxRssi: -35
};

const BEACONS = {
  veranda: {
    label: "BLE веранда",
    x: 251.012,
    y: 180.908
  },
  dancefloor: {
    label: "BLE танцпол",
    x: 965.6,
    y: 318.844
  },
  bar: {
    label: "BLE бар",
    x: 1024.05,
    y: 850.766
  },
  entrance: {
    label: "BLE вход",
    x: 1797.09,
    y: 898.99
  }
};

const ZONES = {
  entrance: {
    label: "Вход / выход",
    x: 1664,
    y: 955,
    radius: 55
  },

  outsideExit: {
    label: "Уход за пределы схемы",
    x: 1664,
    y: 1120,
    radius: 20
  },

  queueLine: {
    label: "Очередь",
    x1: 993,
    y1: 1012.54,
    x2: 1646.21,
    y2: 1012.54
  },

  veranda: {
    label: "Веранда",
    x: 251.012,
    y: 430,
    radius: 170,
    bounds: {
      x: 134.872,
      y: 127.772,
      width: 231.807,
      height: 766.603
    }
  },

  dancefloor: {
    label: "Танцпол",
    x: 965.6,
    y: 430,
    radius: 230,
    bounds: {
      x: 489.939,
      y: 277.196,
      width: 917.712,
      height: 301.033
    }
  },

  bar: {
    label: "Бар",
    x: 1024.05,
    y: 850.766,
    radius: 190,
    bounds: {
      x: 489.939,
      y: 795.966,
      width: 917.712,
      height: 109.599
    }
  },

  toilet: {
    label: "WC / гардероб",
    x: 1675,
    y: 335,
    radius: 120,
    bounds: {
      x: 1547.94,
      y: 91.6079,
      width: 249.886,
      height: 486.621
    }
  },

  stage: {
    label: "Сцена",
    x: 950,
    y: 180,
    radius: 150,
    bounds: {
      x: 595.154,
      y: 128.141,
      width: 707.281,
      height: 109.599
    }
  },

  centralHub: {
    label: "Центральное помещение",
    x: 965,
    y: 670,
    radius: 80
  }
};

const WAYPOINTS = {
  entranceDoor: { x: 1664, y: 955 },
  doorApproach: { x: 1664, y: 1016 },
  outsideLane: { x: 1664, y: 1038 },
  outsideExit: { x: 1664, y: 1120 },
  outsideLeftExit: { x: -70, y: 1038 },
  outsideRightExit: { x: 1990, y: 1038 },

  rightHall: { x: 1640, y: 850 },
  rightPassage: { x: 1486, y: 670 },

  centralHub: { x: 965, y: 670 },

  leftPassage: { x: 438, y: 675 },
  verandaHub: { x: 251, y: 520 },

  toiletHub: { x: 1650, y: 500 },
  barHub: { x: 1024, y: 850 },
  dancefloorHub: { x: 965, y: 430 }
};

const ROUTES = {
  entrance: {
    bar: ["rightHall", "rightPassage", "centralHub", "barHub"],
    dancefloor: ["rightHall", "rightPassage", "centralHub", "dancefloorHub"],
    veranda: ["rightHall", "rightPassage", "centralHub", "leftPassage", "verandaHub"],
    toilet: ["rightHall", "rightPassage", "toiletHub"],
    centralHub: ["rightHall", "rightPassage", "centralHub"]
  },
  bar: {
    bar: ["barHub"],
    dancefloor: ["centralHub", "dancefloorHub"],
    veranda: ["centralHub", "leftPassage", "verandaHub"],
    toilet: ["centralHub", "rightPassage", "toiletHub"],
    outsideExit: ["centralHub", "rightPassage", "rightHall", "entranceDoor", "outsideLane", "outsideExit"]
  },
  dancefloor: {
    bar: ["centralHub", "barHub"],
    dancefloor: ["dancefloorHub"],
    veranda: ["centralHub", "leftPassage", "verandaHub"],
    toilet: ["centralHub", "rightPassage", "toiletHub"],
    outsideExit: ["centralHub", "rightPassage", "rightHall", "entranceDoor", "outsideLane", "outsideExit"]
  },
  veranda: {
    bar: ["verandaHub", "leftPassage", "centralHub", "barHub"],
    dancefloor: ["verandaHub", "leftPassage", "centralHub", "dancefloorHub"],
    veranda: ["verandaHub"],
    toilet: ["verandaHub", "leftPassage", "centralHub", "rightPassage", "toiletHub"],
    outsideExit: ["verandaHub", "leftPassage", "centralHub", "rightPassage", "rightHall", "entranceDoor", "outsideLane", "outsideExit"]
  },
  toilet: {
    bar: ["toiletHub", "rightPassage", "centralHub", "barHub"],
    dancefloor: ["toiletHub", "rightPassage", "centralHub", "dancefloorHub"],
    veranda: ["toiletHub", "rightPassage", "centralHub", "leftPassage", "verandaHub"],
    toilet: ["toiletHub"],
    outsideExit: ["toiletHub", "rightPassage", "rightHall", "entranceDoor", "outsideLane", "outsideExit"]
  },
  centralHub: {
    bar: ["barHub"],
    dancefloor: ["dancefloorHub"],
    veranda: ["leftPassage", "verandaHub"],
    toilet: ["rightPassage", "toiletHub"],
    outsideExit: ["rightPassage", "rightHall", "entranceDoor", "outsideLane", "outsideExit"]
  },
  stage: {
    bar: ["dancefloorHub", "centralHub", "barHub"],
    dancefloor: ["dancefloorHub"],
    veranda: ["dancefloorHub", "centralHub", "leftPassage", "verandaHub"],
    toilet: ["dancefloorHub", "centralHub", "rightPassage", "toiletHub"],
    outsideExit: ["dancefloorHub", "centralHub", "rightPassage", "rightHall", "entranceDoor", "outsideLane", "outsideExit"]
  }
};

const CLUB_STATUSES = new Set(["entering", "moving", "dwelling", "leaving"]);
const VISIBLE_STATUSES = new Set(["approachingQueue", "queue", "entering", "moving", "dwelling", "leaving", "outsideLeaving"]);
const GROUP_LIMIT = 50;
const BEACON_DISPLAY_ORDER = ["bar", "dancefloor", "veranda", "entrance"];
const BLE_TOP_BEACONS_COUNT = 3;
const BLE_DETECTION_ZONES = ["bar", "dancefloor", "veranda", "entrance"];
const OUTSIDE_BLE_LOCATION = "outsideBle";

const WALLS = [
  { x1: 1860.66, y1: 68.2267, x2: 1860.66, y2: 937.715, thickness: 29.2265 },
  { x1: 59.8608, y1: 39.0001, x2: 59.8609, y2: 938.199, thickness: 29.7216 },
  { x1: 1874.97, y1: 53.6134, x2: 73.6102, y2: 53.6132, thickness: 29.2265 },
  { x1: 1603.17, y1: 954.942, x2: 45, y2: 954.942, thickness: 33.8228 },
  { x1: 1875.27, y1: 952.328, x2: 1717.45, y2: 952.328, thickness: 29.2265 },
  { x1: 1486.56, y1: 68.2267, x2: 1486.56, y2: 608.917, thickness: 29.2265 },
  { x1: 1486.56, y1: 730.207, x2: 1486.56, y2: 937.715, thickness: 29.2265 },
  { x1: 438.792, y1: 730.207, x2: 438.792, y2: 937.715, thickness: 29.2265 },
  { x1: 438.792, y1: 68.2267, x2: 438.792, y2: 622.069, thickness: 29.2265 }
];

// 2. DOM-ссылки
const dom = {
  errorBanner: document.getElementById("errorBanner"),
  dataStatus: document.getElementById("dataStatus"),
  mapWrapper: document.getElementById("mapWrapper"),
  clubMap: document.getElementById("clubMap"),
  agentsLayer: document.getElementById("agentsLayer"),
  signalPopup: document.getElementById("signalPopup"),
  debugInfo: document.getElementById("debugInfo"),
  startPauseBtn: document.getElementById("startPauseBtn"),
  resetBtn: document.getElementById("resetBtn"),
  speedRange: document.getElementById("speedRange"),
  speedValue: document.getElementById("speedValue"),
  showBeaconsToggle: document.getElementById("showBeaconsToggle"),
  debugToggle: document.getElementById("debugToggle"),
  clubOccupancy: document.getElementById("clubOccupancy"),
  clubOccupancyHeader: document.getElementById("clubOccupancyHeader"),
  barCount: document.getElementById("barCount"),
  dancefloorCount: document.getElementById("dancefloorCount"),
  verandaCount: document.getElementById("verandaCount"),
  exitedCount: document.getElementById("exitedCount"),
  totalVisitorsCount: document.getElementById("totalVisitorsCount"),
  centralGroupCount: document.getElementById("centralGroupCount"),
  centralGroupHeader: document.getElementById("centralGroupHeader"),
  visibleAgentsCount: document.getElementById("visibleAgentsCount"),
  recommendationsList: document.getElementById("recommendationsList"),
  recommendationGroupSize: document.getElementById("recommendationGroupSize"),
  methodSelect: document.getElementById("methodSelect"),
  backendStatus: document.getElementById("backendStatus"),
  groupLists: {
    centralRoomGroup: document.getElementById("centralRoomList"),
    dancefloorGroup: document.getElementById("dancefloorList"),
    barGroup: document.getElementById("barList"),
    verandaGroup: document.getElementById("verandaList")
  },
  groupMeta: {
    centralRoomGroup: document.getElementById("centralRoomMeta"),
    dancefloorGroup: document.getElementById("dancefloorMeta"),
    barGroup: document.getElementById("barMeta"),
    verandaGroup: document.getElementById("verandaMeta")
  }
};

const state = {
  userIds: [],
  nextUserIndex: 0,
  agents: [],
  nextAgentId: 1,
  selectedAgentId: null,
  running: true,
  dataLoaded: false,
  userIdsDepleted: false,
  speedMultiplier: CONFIG.defaultSpeedMultiplier,
  simulationTime: 0,
  lastTimestamp: 0,
  nextQueueSpawnAt: 0,
  nextAdmissionAt: 0,
  lastUiRenderAt: 0,
  lastRecommendationAt: 0,
  lastRecommendationKey: "",
  recommendations: [],
  exitedCount: 0,
  visitedClubCount: 0,
  cursor: null,
  showBeacons: true,
  debug: false
};

// 3. Загрузка user_id
async function loadUserIds() {
  try {
    const response = await fetch(CONFIG.userIdsUrl, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const text = await response.text();
    state.userIds = text
      .split(/\r?\n/)
      .map((id) => id.trim())
      .filter(Boolean);

    if (state.userIds.length === 0) {
      throw new Error("Файл со списком user_id пуст.");
    }

    state.dataLoaded = true;
    dom.dataStatus.textContent = "Вместимость 300 посетителей.";
    resetSimulation();
  } catch (error) {
    const serverHint =
      window.location.protocol === "file:"
        ? "\n\nЗапустите локальный сервер:\npython -m http.server 8000\nи откройте http://localhost:8000"
        : "";
    showError(`Не удалось прочитать ${CONFIG.userIdsUrl}: ${error.message}.${serverHint}`);
    dom.dataStatus.textContent = "Список user_id не загружен";
  }
}

// 4. Класс/фабрика агента
function createAgent(userId) {
  const queueIndex = getQueuePipelineAgents().length;
  const queueJitter = randomBetween(-5, 5);
  const queuePoint = getQueuePoint(queueIndex, queueJitter);
  const entrySide = Math.random() < 0.5 ? "left" : "right";
  const startPoint = getQueueApproachStart(entrySide, queuePoint);
  const approachPoint = {
    name: "queueApproach",
    x: queuePoint.x + randomBetween(-8, 8),
    y: queuePoint.y + 24 + randomBetween(-4, 4)
  };

  return {
    id: state.nextAgentId++,
    userId,
    x: startPoint.x,
    y: startPoint.y,
    status: "approachingQueue",
    currentZone: "outside",
    targetZone: null,
    route: [approachPoint, { ...queuePoint, name: "queueTail" }],
    routeIndex: 0,
    speed: randomBetween(74, 118),
    dwellTime: 0,
    visitedBar: false,
    verandaVisits: 0,
    dancefloorPreference: Math.random(),
    verandaPreference: Math.random(),
    earlyLeaveProbability: randomBetween(0.03, 0.11),
    bleSignals: {},
    bleTopBeacons: [],
    estimatedLocation: { x: startPoint.x, y: startPoint.y },
    detectedLocation: OUTSIDE_BLE_LOCATION,
    signalColorLocation: OUTSIDE_BLE_LOCATION,
    confidence: 0,
    lastLocationUpdate: 0,
    queueJitter,
    queuePositionIndex: queueIndex,
    entrySide,
    exitSide: Math.random() < 0.5 ? "left" : "right",
    wanderPhase: randomBetween(0, Math.PI * 2),
    wanderFrequency: randomBetween(0.0018, 0.0036),
    wanderStrength: randomBetween(8, 18)
  };
}

function getNextUserId() {
  if (state.nextUserIndex >= state.userIds.length) {
    state.userIdsDepleted = true;
    return null;
  }

  const userId = state.userIds[state.nextUserIndex];
  state.nextUserIndex += 1;
  return userId;
}

function spawnQueueAgent() {
  const userId = getNextUserId();
  if (!userId) {
    return null;
  }

  const agent = createAgent(userId);
  state.agents.push(agent);
  return agent;
}

// 5. Очередь и вход
function fillInitialQueue() {
  const count = Math.min(CONFIG.initialQueueSize, state.userIds.length);
  for (let index = 0; index < count; index += 1) {
    spawnQueueAgent();
  }
  layoutQueue();
}

function maintainQueue() {
  if (state.userIdsDepleted) {
    return;
  }

  if (getQueuePipelineAgents().length >= CONFIG.targetQueueSize) {
    return;
  }

  while (state.simulationTime >= state.nextQueueSpawnAt && getQueuePipelineAgents().length < CONFIG.targetQueueSize) {
    const agent = spawnQueueAgent();
    if (!agent) {
      break;
    }
    state.nextQueueSpawnAt += CONFIG.queueSpawnIntervalMs;
  }
}

function admitFromQueue() {
  if (getClubOccupancy() >= CLUB_CAPACITY) {
    return;
  }

  const queueAgents = getQueueAgents();
  if (queueAgents.length === 0) {
    return;
  }

  const availableSlots = CLUB_CAPACITY - getClubOccupancy();
  const admissions = Math.min(availableSlots, 1);

  for (let count = 0; count < admissions; count += 1) {
    const agent = queueAgents[count];
    if (!agent) {
      return;
    }

    agent.status = "entering";
    agent.currentZone = "entrance";
    agent.targetZone = "entrance";
    agent.route = addChaoticRoutePoints([
      jitterNamedPoint("doorApproach", WAYPOINTS.doorApproach, 12),
      namedPoint("entranceDoor", WAYPOINTS.entranceDoor)
    ]);
    agent.routeIndex = 0;
    state.visitedClubCount += 1;
  }
}

function processAdmissions() {
  while (state.simulationTime >= state.nextAdmissionAt) {
    admitFromQueue();
    state.nextAdmissionAt += CONFIG.admissionIntervalMs;
    if (getClubOccupancy() >= CLUB_CAPACITY || getQueueAgents().length === 0) {
      break;
    }
  }
}

function layoutQueue() {
  const queueAgents = getQueueAgents();
  queueAgents.forEach((agent, index) => {
    agent.queuePositionIndex = index;

    if (agent.status !== "queue") {
      return;
    }

    const point = getQueuePoint(index, agent.queueJitter);
    agent.x = point.x;
    agent.y = point.y;
  });
}

function getQueuePoint(index, jitter = 0) {
  const line = ZONES.queueLine;
  const perRow = 62;
  const row = Math.floor(index / perRow);
  const position = index % perRow;
  const t = perRow === 1 ? 0 : position / (perRow - 1);

  return {
    x: lerp(line.x2, line.x1, t),
    y: line.y1 - 22 + row * 20 + jitter
  };
}

function getQueueAgents() {
  return state.agents.filter((agent) => agent.status === "queue");
}

function getQueuePipelineAgents() {
  return state.agents.filter((agent) => agent.status === "queue" || agent.status === "approachingQueue");
}

function getQueueApproachStart(side, queuePoint) {
  return {
    x: side === "left" ? -70 : CONFIG.viewBox.width + 70,
    y: queuePoint.y + 24 + randomBetween(-8, 8)
  };
}

// 6. Поведение и переходы
function chooseNextZone(agent) {
  if (!agent.visitedBar) {
    return canEnterZone("bar") || agent.currentZone === "bar" ? "bar" : chooseAvailableZone(["dancefloor", "veranda", "toilet"]);
  }

  const choices = [];

  addWeightedChoice(choices, "bar", agent.currentZone === "bar" ? 14 : 20);
  addWeightedChoice(choices, "dancefloor", 24 + Math.round(agent.dancefloorPreference * 38));
  addWeightedChoice(choices, "veranda", 10 + Math.round(agent.verandaPreference * 34));
  addWeightedChoice(choices, "toilet", agent.currentZone === "toilet" ? 2 : 6);

  if (agent.currentZone === "dancefloor") {
    addWeightedChoice(choices, "dancefloor", 15 + Math.round(agent.dancefloorPreference * 22));
  }

  if (agent.currentZone === "veranda" && agent.verandaPreference > 0.55 && agent.verandaVisits < 4) {
    addWeightedChoice(choices, "veranda", 14);
  }

  const preferred = pickWeighted(choices);
  if (canEnterZone(preferred)) {
    return preferred;
  }

  return chooseAvailableZone(["dancefloor", "bar", "veranda", "toilet"]);
}

function canEnterZone(zoneName) {
  if (zoneName === "veranda") {
    const limit = Math.max(1, Math.floor(getClubOccupancy() * MAX_VERANDA_SHARE));
    return getZoneOccupancy("veranda") < limit;
  }

  if (zoneName === "toilet") {
    const limit = Math.max(1, Math.floor(getClubOccupancy() * MAX_TOILET_SHARE));
    return getZoneOccupancy("toilet") < limit;
  }

  return true;
}

function getZoneOccupancy(zoneName) {
  return state.agents.filter((agent) => {
    if (!CLUB_STATUSES.has(agent.status)) {
      return false;
    }
    if (agent.status === "moving" && agent.targetZone === zoneName) {
      return true;
    }
    return agent.currentZone === zoneName && agent.status !== "leaving";
  }).length;
}

function startLeaving(agent) {
  if (!agent.visitedBar || agent.status === "leaving" || agent.status === "outsideLeaving") {
    return false;
  }

  agent.status = "leaving";
  agent.targetZone = "outsideExit";
  agent.exitSide = Math.random() < 0.5 ? "left" : "right";
  agent.route = buildRoute(agent.currentZone, "outsideExit").map((point) => {
    if (point.name !== "outsideExit") {
      return point;
    }

    const exitPoint = agent.exitSide === "left" ? WAYPOINTS.outsideLeftExit : WAYPOINTS.outsideRightExit;
    return jitterNamedPoint(point.name, exitPoint, 18);
  });
  agent.routeIndex = 0;
  agent.dwellTime = 0;
  return true;
}

function shouldLeave(agent) {
  if (!agent.visitedBar) {
    return false;
  }

  let probability = agent.earlyLeaveProbability;

  if (agent.currentZone === "toilet") {
    probability *= 0.3;
  }

  if (agent.currentZone === "dancefloor" && agent.dancefloorPreference > 0.62) {
    probability *= 0.45;
  }

  if (agent.currentZone === "veranda" && agent.verandaPreference > 0.7 && agent.verandaVisits < 3) {
    probability *= 0.55;
  }

  return Math.random() < probability;
}

function arriveAtZone(agent, zoneName) {
  agent.status = "dwelling";
  agent.currentZone = zoneName;
  agent.targetZone = null;
  agent.route = [];
  agent.routeIndex = 0;
  agent.dwellTime = getDwellTime(agent, zoneName);

  if (zoneName === "bar") {
    agent.visitedBar = true;
  }

  if (zoneName === "veranda") {
    agent.verandaVisits += 1;
  }
}

function getDwellTime(agent, zoneName) {
  if (zoneName === "bar") {
    return randomBetween(10000, 22000);
  }

  if (zoneName === "dancefloor") {
    return randomBetween(11000, 24000) + agent.dancefloorPreference * 18000;
  }

  if (zoneName === "veranda") {
    return randomBetween(9000, 21000) + agent.verandaPreference * 12000;
  }

  if (zoneName === "toilet") {
    return randomBetween(3500, 8500);
  }

  return randomBetween(5500, 12000);
}

function chooseAvailableZone(zoneNames) {
  const available = zoneNames.filter((zoneName) => canEnterZone(zoneName));
  if (available.length > 0) {
    return available[Math.floor(Math.random() * available.length)];
  }

  return "bar";
}

function addWeightedChoice(choices, zoneName, weight) {
  choices.push({ zoneName, weight: Math.max(1, weight) });
}

function pickWeighted(choices) {
  const total = choices.reduce((sum, choice) => sum + choice.weight, 0);
  let cursor = Math.random() * total;

  for (const choice of choices) {
    cursor -= choice.weight;
    if (cursor <= 0) {
      return choice.zoneName;
    }
  }

  return choices[choices.length - 1].zoneName;
}

// 7. Маршруты и движение
function buildRoute(fromZone, toZone) {
  const normalizedFrom = normalizeRouteZone(fromZone);
  const normalizedTo = normalizeRouteZone(toZone);
  const routeNames = getRouteNames(normalizedFrom, normalizedTo);
  const points = routeNames.map((name) => namedPoint(name, WAYPOINTS[name]));

  if (normalizedTo === "outsideExit") {
    return points;
  }

  if (ZONES[normalizedTo]) {
    points.push({ ...randomPointInZone(normalizedTo), name: normalizedTo });
  }

  return addChaoticRoutePoints(points);
}

function getRouteNames(fromZone, toZone) {
  if (ROUTES[fromZone] && ROUTES[fromZone][toZone]) {
    return ROUTES[fromZone][toZone];
  }

  if (fromZone !== "centralHub" && ROUTES[fromZone] && ROUTES[fromZone].outsideExit && toZone !== "outsideExit") {
    return [...ROUTES[fromZone].outsideExit.slice(0, -2), ...(ROUTES.centralHub[toZone] || [])];
  }

  if (ROUTES.centralHub[toZone]) {
    return ROUTES.centralHub[toZone];
  }

  if (toZone === "outsideExit") {
    return ROUTES.centralHub.outsideExit;
  }

  return ["centralHub"];
}

function normalizeRouteZone(zoneName) {
  if (!zoneName || zoneName === "queue" || zoneName === "outside") {
    return "entrance";
  }

  if (zoneName === "outsideExit") {
    return "outsideExit";
  }

  return ROUTES[zoneName] ? zoneName : "centralHub";
}

function namedPoint(name, point) {
  return { name, x: point.x, y: point.y };
}

function jitterNamedPoint(name, point, amount) {
  return {
    name,
    x: point.x + randomBetween(-amount, amount),
    y: point.y + randomBetween(-amount, amount)
  };
}

function addChaoticRoutePoints(points) {
  if (points.length < 2) {
    return points;
  }

  const route = [points[0]];

  for (let index = 1; index < points.length; index += 1) {
    const start = points[index - 1];
    const end = points[index];
    const distance = Math.hypot(end.x - start.x, end.y - start.y);

    if (distance > CONFIG.routeChaosMinSegment) {
      const segmentCount = clamp(Math.floor(distance / 230), 1, 3);
      for (let step = 1; step <= segmentCount; step += 1) {
        const t = step / (segmentCount + 1);
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const length = Math.max(1, Math.hypot(dx, dy));
        const offset = randomBetween(-CONFIG.routeChaosOffset, CONFIG.routeChaosOffset);

        route.push({
          name: `drift:${end.name || index}:${step}`,
          x: lerp(start.x, end.x, t) + (-dy / length) * offset + randomBetween(-8, 8),
          y: lerp(start.y, end.y, t) + (dx / length) * offset + randomBetween(-8, 8)
        });
      }
    }

    route.push(end);
  }

  return route;
}

function moveAgent(agent, deltaMs) {
  if (!agent.route || agent.routeIndex >= agent.route.length) {
    finishRoute(agent);
    return;
  }

  let remainingDistance = (agent.speed * deltaMs) / 1000;

  while (remainingDistance > 0 && agent.routeIndex < agent.route.length) {
    const target = agent.route[agent.routeIndex];
    const dx = target.x - agent.x;
    const dy = target.y - agent.y;
    const distance = Math.hypot(dx, dy);

    if (distance <= CONFIG.routeArrivalDistance || distance <= remainingDistance) {
      agent.x = target.x;
      agent.y = target.y;
      remainingDistance -= distance;
      handleReachedRoutePoint(agent, target);
    } else {
      const lateralStep =
        Math.sin(state.simulationTime * agent.wanderFrequency + agent.wanderPhase) *
        agent.wanderStrength *
        (deltaMs / 1000) *
        Math.min(1, distance / 80);

      agent.x += (dx / distance) * remainingDistance + (-dy / distance) * lateralStep;
      agent.y += (dy / distance) * remainingDistance + (dx / distance) * lateralStep;
      remainingDistance = 0;
    }
  }

  if (
    agent.status === "outsideLeaving" &&
    (agent.y > CONFIG.viewBox.height + 18 || agent.x < -18 || agent.x > CONFIG.viewBox.width + 18)
  ) {
    markExited(agent);
  }
}

function handleReachedRoutePoint(agent, point) {
  if (agent.status === "leaving" && point.name === "entranceDoor") {
    agent.status = "outsideLeaving";
    agent.currentZone = "outside";
  }

  agent.routeIndex += 1;

  if (agent.routeIndex >= agent.route.length) {
    finishRoute(agent);
  }
}

function finishRoute(agent) {
  if (agent.status === "approachingQueue") {
    agent.status = "queue";
    agent.currentZone = "queue";
    agent.targetZone = null;
    agent.route = [];
    agent.routeIndex = 0;
    layoutQueue();
    return;
  }

  if (agent.status === "entering") {
    const nextZone = chooseNextZone(agent);
    startMovingToZone(agent, nextZone, "entrance");
    return;
  }

  if (agent.status === "moving") {
    arriveAtZone(agent, agent.targetZone);
    return;
  }

  if (agent.status === "leaving" || agent.status === "outsideLeaving") {
    markExited(agent);
  }
}

function startMovingToZone(agent, zoneName, fromZone = agent.currentZone) {
  agent.status = "moving";
  agent.targetZone = zoneName;
  agent.route = buildRoute(fromZone, zoneName);
  agent.routeIndex = 0;
}

function randomPointInZone(zoneName) {
  const zone = ZONES[zoneName];

  if (!zone) {
    return { ...WAYPOINTS.centralHub };
  }

  if (zoneName === "outsideExit") {
    return {
      x: zone.x + randomBetween(-12, 12),
      y: zone.y + randomBetween(8, 34)
    };
  }

  if (zone.bounds) {
    const padding = 22;
    return {
      x: randomBetween(zone.bounds.x + padding, zone.bounds.x + zone.bounds.width - padding),
      y: randomBetween(zone.bounds.y + padding, zone.bounds.y + zone.bounds.height - padding)
    };
  }

  const angle = Math.random() * Math.PI * 2;
  const radius = Math.sqrt(Math.random()) * zone.radius * 0.65;
  return {
    x: zone.x + Math.cos(angle) * radius,
    y: zone.y + Math.sin(angle) * radius
  };
}

function updateDwellingAgent(agent, deltaMs) {
  updateZoneMicroMovement(agent, deltaMs);
  agent.dwellTime -= deltaMs;
  if (agent.dwellTime > 0) {
    return;
  }

  if (shouldLeave(agent) && startLeaving(agent)) {
    return;
  }

  const nextZone = chooseNextZone(agent);
  startMovingToZone(agent, nextZone);
}

function updateZoneMicroMovement(agent, deltaMs) {
  if (agent.currentZone !== "dancefloor") {
    return;
  }

  const zone = ZONES.dancefloor;
  const pulse = state.simulationTime * agent.wanderFrequency + agent.wanderPhase;
  const step = (deltaMs / 1000) * (4 + agent.dancefloorPreference * 8);

  agent.x += Math.cos(pulse * 1.7) * step + randomBetween(-0.6, 0.6);
  agent.y += Math.sin(pulse * 1.3) * step + randomBetween(-0.6, 0.6);

  if (zone.bounds) {
    const padding = 24;
    agent.x = clamp(agent.x, zone.bounds.x + padding, zone.bounds.x + zone.bounds.width - padding);
    agent.y = clamp(agent.y, zone.bounds.y + padding, zone.bounds.y + zone.bounds.height - padding);
  }
}

function markExited(agent) {
  if (agent.status === "exited") {
    return;
  }

  agent.status = "exited";
  agent.currentZone = "outside";
  agent.targetZone = null;
  agent.route = [];
  agent.routeIndex = 0;
  state.exitedCount += 1;

  if (state.selectedAgentId === agent.id) {
    state.selectedAgentId = null;
  }
}

// 8. BLE-расчеты
function updateBleForAgent(agent) {
  const signals = {};

  for (const [beaconName, beacon] of Object.entries(BEACONS)) {
    if (isSignalBlockedByWall(agent, beacon)) {
      signals[beaconName] = null;
      continue;
    }

    const distance = Math.max(1, Math.hypot(agent.x - beacon.x, agent.y - beacon.y));
    const noise = randomBetween(BLE_CONFIG.noiseMin, BLE_CONFIG.noiseMax);
    const rssi =
      BLE_CONFIG.txPower -
      10 * BLE_CONFIG.pathLossExponent * Math.log10(distance / BLE_CONFIG.referenceDistance) +
      noise;
    signals[beaconName] = clamp(Math.round(rssi), BLE_CONFIG.minRssi, BLE_CONFIG.maxRssi);
  }

  const sorted = Object.entries(signals)
    .filter(([, rssi]) => Number.isFinite(rssi))
    .sort((a, b) => b[1] - a[1]);

  if (sorted.length === 0 || sorted[0][1] <= BLE_CONFIG.minRssi + 2) {
    agent.bleSignals = signals;
    agent.bleTopBeacons = [];
    agent.estimatedLocation = { x: agent.x, y: agent.y };
    agent.detectedLocation = OUTSIDE_BLE_LOCATION;
    agent.signalColorLocation = OUTSIDE_BLE_LOCATION;
    agent.confidence = 0;
    agent.lastLocationUpdate = state.simulationTime;
    return;
  }

  const topBeacons = sorted.slice(0, BLE_TOP_BEACONS_COUNT);
  const weightedLocation = estimateLocationFromTopBeacons(topBeacons);
  const detected = detectLocationFromTopBeaconFingerprint(topBeacons, weightedLocation);

  agent.bleSignals = signals;
  agent.bleTopBeacons = topBeacons.map(([name, rssi]) => ({ name, rssi }));
  agent.estimatedLocation = detected.point;
  agent.detectedLocation = detected.name;
  agent.signalColorLocation = sorted[0][0];
  agent.confidence = sorted.length > 1 ? sorted[0][1] - sorted[1][1] : 0;
  agent.lastLocationUpdate = state.simulationTime;
}

function estimateLocationFromTopBeacons(topBeacons) {
  let weightedX = 0;
  let weightedY = 0;
  let totalWeight = 0;

  for (const [beaconName, rssi] of topBeacons) {
    const beacon = BEACONS[beaconName];
    const estimatedDistance =
      BLE_CONFIG.referenceDistance *
      10 ** ((BLE_CONFIG.txPower - rssi) / (10 * BLE_CONFIG.pathLossExponent));
    const weight = 1 / Math.max(1, estimatedDistance * estimatedDistance);

    weightedX += beacon.x * weight;
    weightedY += beacon.y * weight;
    totalWeight += weight;
  }

  if (totalWeight === 0) {
    return { ...WAYPOINTS.centralHub };
  }

  return {
    x: weightedX / totalWeight,
    y: weightedY / totalWeight
  };
}

function detectLocationFromTopBeaconFingerprint(topBeacons, weightedLocation) {
  if (topBeacons.length === 0) {
    return {
      name: OUTSIDE_BLE_LOCATION,
      point: weightedLocation,
      score: 0
    };
  }

  let bestMatch = {
    name: topBeacons[0][0],
    point: weightedLocation,
    score: Infinity
  };

  for (const zoneName of BLE_DETECTION_ZONES) {
    for (const sample of getDetectionSamples(zoneName)) {
      let score = Math.hypot(sample.x - weightedLocation.x, sample.y - weightedLocation.y) / 180;

      for (const [beaconName, observedRssi] of topBeacons) {
        const expectedRssi = getExpectedRssiAtPoint(sample, BEACONS[beaconName]);
        score += (expectedRssi - observedRssi) ** 2;
      }

      if (score < bestMatch.score) {
        bestMatch = {
          name: zoneName,
          point: sample,
          score
        };
      }
    }
  }

  return bestMatch;
}

function getDetectionSamples(zoneName) {
  if (zoneName === "entrance") {
    return [
      namedPoint("entranceDoor", WAYPOINTS.entranceDoor),
      namedPoint("entranceBeacon", BEACONS.entrance),
      { name: "entranceMidpoint", x: 1730, y: 930 }
    ];
  }

  const zone = ZONES[zoneName];
  if (!zone || !zone.bounds) {
    return [{ name: zoneName, x: zone.x, y: zone.y }];
  }

  const samples = [];
  for (const fx of [0.18, 0.5, 0.82]) {
    for (const fy of [0.22, 0.5, 0.78]) {
      samples.push({
        name: `${zoneName}:${fx}:${fy}`,
        x: zone.bounds.x + zone.bounds.width * fx,
        y: zone.bounds.y + zone.bounds.height * fy
      });
    }
  }

  return samples;
}

function getExpectedRssiAtPoint(point, beacon) {
  if (isSignalBlockedByWall(point, beacon)) {
    return BLE_CONFIG.minRssi;
  }

  const distance = Math.max(1, Math.hypot(point.x - beacon.x, point.y - beacon.y));
  const rssi =
    BLE_CONFIG.txPower -
    10 * BLE_CONFIG.pathLossExponent * Math.log10(distance / BLE_CONFIG.referenceDistance);
  return clamp(Math.round(rssi), BLE_CONFIG.minRssi, BLE_CONFIG.maxRssi);
}

function isSignalBlockedByWall(from, to) {
  return WALLS.some((wall) => segmentDistance(from, to, { x: wall.x1, y: wall.y1 }, { x: wall.x2, y: wall.y2 }) <= wall.thickness / 2);
}

function segmentDistance(a, b, c, d) {
  if (segmentsIntersect(a, b, c, d)) {
    return 0;
  }

  return Math.min(
    pointToSegmentDistance(a, c, d),
    pointToSegmentDistance(b, c, d),
    pointToSegmentDistance(c, a, b),
    pointToSegmentDistance(d, a, b)
  );
}

function pointToSegmentDistance(point, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;

  if (lengthSquared === 0) {
    return Math.hypot(point.x - start.x, point.y - start.y);
  }

  const t = clamp(((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared, 0, 1);
  return Math.hypot(point.x - (start.x + dx * t), point.y - (start.y + dy * t));
}

function segmentsIntersect(a, b, c, d) {
  const o1 = orientation(a, b, c);
  const o2 = orientation(a, b, d);
  const o3 = orientation(c, d, a);
  const o4 = orientation(c, d, b);

  if (o1 !== o2 && o3 !== o4) {
    return true;
  }

  return (
    (o1 === 0 && pointOnSegment(c, a, b)) ||
    (o2 === 0 && pointOnSegment(d, a, b)) ||
    (o3 === 0 && pointOnSegment(a, c, d)) ||
    (o4 === 0 && pointOnSegment(b, c, d))
  );
}

function orientation(a, b, c) {
  const value = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
  if (Math.abs(value) < 0.0001) {
    return 0;
  }
  return value > 0 ? 1 : 2;
}

function pointOnSegment(point, start, end) {
  return (
    point.x <= Math.max(start.x, end.x) &&
    point.x >= Math.min(start.x, end.x) &&
    point.y <= Math.max(start.y, end.y) &&
    point.y >= Math.min(start.y, end.y)
  );
}

function updateBleSignals(force = false) {
  for (const agent of state.agents) {
    if (!VISIBLE_STATUSES.has(agent.status)) {
      continue;
    }

    if (force || state.simulationTime - agent.lastLocationUpdate >= CONFIG.bleUpdateIntervalMs) {
      updateBleForAgent(agent);
    }
  }
}

// 9. Группы рекомендаций
function getRecommendationGroups() {
  const inClubAgents = state.agents.filter((agent) => CLUB_STATUSES.has(agent.status));
  const centralRoomGroup = inClubAgents
    .filter((agent) => agent.detectedLocation === "bar" || agent.detectedLocation === "dancefloor")
    .map((agent) => agent.userId);
  const dancefloorGroup = inClubAgents
    .filter((agent) => agent.detectedLocation === "dancefloor")
    .map((agent) => agent.userId);
  const barGroup = inClubAgents
    .filter((agent) => agent.detectedLocation === "bar")
    .map((agent) => agent.userId);
  const verandaGroup = inClubAgents
    .filter((agent) => agent.detectedLocation === "veranda")
    .map((agent) => agent.userId);

  return {
    centralRoomGroup,
    dancefloorGroup,
    barGroup,
    verandaGroup
  };
}

function getGroupAgents(groupName) {
  const inClubAgents = state.agents.filter((agent) => CLUB_STATUSES.has(agent.status));

  if (groupName === "centralRoomGroup") {
    return inClubAgents.filter((agent) => agent.detectedLocation === "bar" || agent.detectedLocation === "dancefloor");
  }

  if (groupName === "dancefloorGroup") {
    return inClubAgents.filter((agent) => agent.detectedLocation === "dancefloor");
  }

  if (groupName === "barGroup") {
    return inClubAgents.filter((agent) => agent.detectedLocation === "bar");
  }

  if (groupName === "verandaGroup") {
    return inClubAgents.filter((agent) => agent.detectedLocation === "veranda");
  }

  return [];
}

// 10. Треки-кандидаты
function getMockRecommendations(userIds) {
  const source = userIds.length > 0 ? [...userIds].sort().join("|") : "empty-central-room-group";
  let seed = hashString(source);
  const tracks = [];

  for (let index = 0; index < CONFIG.recommendationTopN; index += 1) {
    seed = seededStep(seed + index * 97);
    const trackNumber = 100000 + (seed % 900000);
    tracks.push({ trackId: String(trackNumber), score: null });
  }

  return tracks;
}

async function fetchRecommendationsFromBackend(userIds) {
  try {
    const response = await fetch(CONFIG.backendUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_ids: userIds,
        method: CONFIG.recommendationMethod,
        top_n: CONFIG.recommendationTopN
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    setBackendStatus(true, payload.method);
    return (payload.tracks || []).map((track) => ({
      trackId: String(track.track_id),
      score: typeof track.score === "number" ? track.score : null
    }));
  } catch (error) {
    setBackendStatus(false);
    return getMockRecommendations(userIds);
  }
}

function setBackendStatus(online, method) {
  if (!dom.backendStatus) {
    return;
  }
  if (online) {
    dom.backendStatus.textContent = `backend: ${method || CONFIG.recommendationMethod}`;
    dom.backendStatus.classList.remove("offline");
  } else {
    dom.backendStatus.textContent = "backend offline · mock";
    dom.backendStatus.classList.add("offline");
  }
}

async function updateRecommendations(force = false) {
  const groups = getRecommendationGroups();
  const userIds = groups.centralRoomGroup;
  const key = `${CONFIG.recommendationMethod}::${userIds.slice().sort().join("|")}`;

  if (!force && key === state.lastRecommendationKey && state.recommendations.length > 0) {
    return;
  }

  state.lastRecommendationKey = key;
  state.recommendations = await fetchRecommendationsFromBackend(userIds);
}

// 11. Рендеринг
function renderMap() {
  const parts = [];

  parts.push(renderOverlay());
  parts.push(renderAgents());

  dom.agentsLayer.innerHTML = parts.join("");
}

function renderOverlay() {
  const parts = [];

  if (state.showBeacons || state.debug) {
    for (const [name, beacon] of Object.entries(BEACONS)) {
      parts.push(`
        <g class="beacon" data-beacon="${name}">
          <circle class="beacon-ring" cx="${beacon.x}" cy="${beacon.y}" r="24"></circle>
          <circle class="beacon-core" cx="${beacon.x}" cy="${beacon.y}" r="9"></circle>
        </g>
      `);
    }
  }

  if (state.debug) {
    for (const [name, zone] of Object.entries(ZONES)) {
      if (name === "queueLine") {
        continue;
      }
      parts.push(`<circle class="zone-center" cx="${zone.x}" cy="${zone.y}" r="12"></circle>`);
    }

    const selected = getSelectedAgent();
    if (selected && selected.route && selected.route.length > selected.routeIndex) {
      const routePoints = [{ x: selected.x, y: selected.y, name: "agent" }, ...selected.route.slice(selected.routeIndex)];
      const polyline = routePoints.map((point) => `${point.x},${point.y}`).join(" ");
      parts.push(`<polyline class="debug-route" points="${polyline}"></polyline>`);
      for (const point of routePoints.slice(1)) {
        parts.push(`<circle class="debug-waypoint" cx="${point.x}" cy="${point.y}" r="8"></circle>`);
      }
    }
  }

  return parts.join("");
}

function renderAgents() {
  return state.agents
    .filter((agent) => {
      if (!VISIBLE_STATUSES.has(agent.status)) {
        return false;
      }
      if (agent.status === "queue" && agent.queuePositionIndex >= CONFIG.visibleQueueLimit) {
        return false;
      }
      return true;
    })
    .map((agent) => {
      const selected = state.selectedAgentId === agent.id;
      const radius = selected ? CONFIG.selectedAgentRadius : CONFIG.agentRadius;
      const classNames = ["agent-dot", getAgentColorClass(agent), selected ? "agent-selected" : ""]
        .filter(Boolean)
        .join(" ");
      return `
        <g class="agent-node" data-agent-id="${agent.id}">
          <title>${escapeHtml(agent.userId)} · ${escapeHtml(agent.detectedLocation)} · ${escapeHtml(agent.status)}</title>
          <circle
            class="${classNames}"
            cx="${agent.x.toFixed(2)}"
            cy="${agent.y.toFixed(2)}"
            r="${radius}"
          ></circle>
          <circle
            class="agent-hit"
            cx="${agent.x.toFixed(2)}"
            cy="${agent.y.toFixed(2)}"
            r="24"
          ></circle>
        </g>
      `;
    })
    .join("");
}

function getAgentColorClass(agent) {
  return `agent-${agent.signalColorLocation || OUTSIDE_BLE_LOCATION}`;
}

function renderUi() {
  const groups = getRecommendationGroups();
  const clubOccupancy = getClubOccupancy();
  const visibleAgentsCount = state.agents.filter((agent) => VISIBLE_STATUSES.has(agent.status)).length;

  setText(dom.clubOccupancy, clubOccupancy);
  setText(dom.clubOccupancyHeader, clubOccupancy);
  setText(dom.barCount, groups.barGroup.length);
  setText(dom.dancefloorCount, groups.dancefloorGroup.length);
  setText(dom.verandaCount, groups.verandaGroup.length);
  setText(dom.exitedCount, state.exitedCount);
  setText(dom.totalVisitorsCount, state.visitedClubCount);
  setText(dom.centralGroupCount, groups.centralRoomGroup.length);
  setText(dom.centralGroupHeader, groups.centralRoomGroup.length);
  dom.visibleAgentsCount.textContent = `${visibleAgentsCount} visible`;

  renderSignalPopup();
  renderGroupList("centralRoomGroup");
  renderGroupList("dancefloorGroup");
  renderGroupList("barGroup");
  renderGroupList("verandaGroup");
  renderRecommendations(groups.centralRoomGroup.length);
  renderDebugInfo(visibleAgentsCount);

  if (state.userIdsDepleted) {
    dom.dataStatus.textContent = `user_id закончились. Симуляция продолжается для активных посетителей`;
  }
}

function renderGroupList(groupName) {
  const agents = getGroupAgents(groupName);
  const shown = agents.slice(0, GROUP_LIMIT);
  const meta = agents.length > GROUP_LIMIT ? `Показаны первые ${GROUP_LIMIT} из ${agents.length}` : `${agents.length} пользователей`;
  dom.groupMeta[groupName].textContent = meta;

  if (shown.length === 0) {
    dom.groupLists[groupName].innerHTML = `<div class="empty-state">Группа пока пуста</div>`;
    return;
  }

  dom.groupLists[groupName].innerHTML = shown
    .map((agent) => {
      const selected = state.selectedAgentId === agent.id ? " selected" : "";
      return `
        <button class="user-row${selected}" type="button" data-agent-id="${agent.id}">
          <b>${escapeHtml(agent.userId)}</b>
        </button>
      `;
    })
    .join("");
}

function renderSignalPopup() {
  const agent = getSelectedAgent();

  if (!agent) {
    dom.signalPopup.hidden = true;
    return;
  }

  if (!VISIBLE_STATUSES.has(agent.status)) {
    dom.signalPopup.hidden = true;
    return;
  }

  dom.signalPopup.hidden = false;
  dom.signalPopup.innerHTML = `
    <b>uid: ${escapeHtml(agent.userId)}</b>
    <dl>
      ${BEACON_DISPLAY_ORDER.map((name) => `<dt>${escapeHtml(getBeaconShortLabel(name))}</dt><dd>${formatSignal(agent.bleSignals[name])}</dd>`).join("")}
      <dt>detected</dt><dd>${escapeHtml(getLocationLabel(agent.detectedLocation))}</dd>
      <dt>confidence</dt><dd>${agent.confidence} dB</dd>
    </dl>
  `;
  positionSignalPopup(agent);
}

function positionSignalPopup(agent) {
  const wrapperRect = dom.mapWrapper.getBoundingClientRect();
  const popupWidth = dom.signalPopup.offsetWidth || 240;
  const popupHeight = dom.signalPopup.offsetHeight || 140;
  const x = (agent.x / CONFIG.viewBox.width) * wrapperRect.width + 12;
  const y = (agent.y / CONFIG.viewBox.height) * wrapperRect.height - popupHeight - 12;

  dom.signalPopup.style.left = `${clamp(x, 8, wrapperRect.width - popupWidth - 8)}px`;
  dom.signalPopup.style.top = `${clamp(y, 8, wrapperRect.height - popupHeight - 8)}px`;
}

function renderRecommendations(groupSize) {
  dom.recommendationGroupSize.textContent = `${groupSize} uid`;

  if (state.recommendations.length === 0) {
    dom.recommendationsList.innerHTML = `<li class="empty-state">Группа в зале пуста</li>`;
    return;
  }

  dom.recommendationsList.innerHTML = state.recommendations
    .map((track) => {
      const score = track.score === null || track.score === undefined
        ? ""
        : `<span class="reco-score">${track.score.toFixed(3)}</span>`;
      return `<li><span class="reco-track">track_${escapeHtml(track.trackId)}</span>${score}</li>`;
    })
    .join("");
}

function renderDebugInfo(visibleAgentsCount) {
  dom.debugInfo.hidden = !state.debug;
  if (!state.debug) {
    return;
  }

  const cursor = state.cursor ? `${Math.round(state.cursor.x)}, ${Math.round(state.cursor.y)}` : "-,-";
  dom.debugInfo.textContent = `viewBox: 0 0 1920 1080 · cursor: ${cursor} · active agents: ${visibleAgentsCount}`;
}

// 12. UI и event listeners
function setupEventListeners() {
  dom.clubMap.addEventListener("error", () => {
    showError(`Не найден файл ${CONFIG.mapUrl}. Проверьте, что club_map.svg лежит в папке assets/.`);
  });

  dom.startPauseBtn.addEventListener("click", () => {
    state.running = !state.running;
    dom.startPauseBtn.textContent = state.running ? "Пауза" : "Старт";
  });

  dom.resetBtn.addEventListener("click", () => {
    resetSimulation();
  });

  dom.speedRange.addEventListener("input", () => {
    state.speedMultiplier = Number(dom.speedRange.value);
    dom.speedValue.textContent = `${state.speedMultiplier.toFixed(2)}x`;
  });

  if (dom.showBeaconsToggle) {
    dom.showBeaconsToggle.addEventListener("change", () => {
      state.showBeacons = dom.showBeaconsToggle.checked;
    });
  }

  if (dom.debugToggle) {
    dom.debugToggle.addEventListener("change", () => {
      state.debug = dom.debugToggle.checked;
    });
  }

  if (dom.methodSelect) {
    dom.methodSelect.value = CONFIG.recommendationMethod;
    dom.methodSelect.addEventListener("change", () => {
      CONFIG.recommendationMethod = dom.methodSelect.value;
      updateRecommendations(true).then(() => renderUi());
    });
  }

  dom.mapWrapper.addEventListener("pointerdown", (event) => {
    const target =
      event.target instanceof Element
        ? event.target.closest("[data-agent-id]")
        : null;
    const clickedPoint = clientToSvgPoint(event);
    const nearestAgent = target ? null : getNearestVisibleAgent(clickedPoint);

    if (!target && !nearestAgent) {
      dom.signalPopup.hidden = true;
      state.selectedAgentId = null;
      renderMap();
      renderUi();
      return;
    }

    event.preventDefault();
    selectAgent(target ? Number(target.dataset.agentId) : nearestAgent.id);
  });

  document.addEventListener("click", (event) => {
    const target = event.target.closest(".user-row[data-agent-id]");
    if (!target) {
      return;
    }
    selectAgent(Number(target.dataset.agentId));
  });

  dom.mapWrapper.addEventListener("mousemove", (event) => {
    state.cursor = clientToSvgPoint(event);
  });

  dom.mapWrapper.addEventListener("mouseleave", () => {
    state.cursor = null;
  });
}

function resetSimulation() {
  state.nextUserIndex = 0;
  state.agents = [];
  state.nextAgentId = 1;
  state.selectedAgentId = null;
  state.userIdsDepleted = false;
  state.simulationTime = 0;
  state.nextQueueSpawnAt = CONFIG.queueSpawnIntervalMs;
  state.nextAdmissionAt = 0;
  state.lastRecommendationAt = 0;
  state.lastRecommendationKey = "";
  state.recommendations = [];
  state.exitedCount = 0;
  state.visitedClubCount = 0;
  state.speedMultiplier = Number(dom.speedRange.value);

  fillInitialQueue();
  updateBleSignals(true);
  updateRecommendations(true);
  renderMap();
  renderUi();
}

function selectAgent(agentId) {
  const agent = state.agents.find((candidate) => candidate.id === agentId);

  if (!agent || !VISIBLE_STATUSES.has(agent.status)) {
    state.selectedAgentId = null;
    dom.signalPopup.hidden = true;
    renderMap();
    renderUi();
    return;
  }

  updateBleForAgent(agent);
  state.selectedAgentId = agent.id;
  renderMap();
  renderUi();
}

// 13. Main loop
function loop(timestamp) {
  if (!state.lastTimestamp) {
    state.lastTimestamp = timestamp;
  }

  const frameDeltaMs = Math.min(timestamp - state.lastTimestamp, 100);
  state.lastTimestamp = timestamp;

  if (state.running && state.dataLoaded) {
    const simulationDeltaMs = frameDeltaMs * state.speedMultiplier;
    state.simulationTime += simulationDeltaMs;
    updateSimulation(simulationDeltaMs);
  }

  if (state.running) {
    renderMap();
  }

  if (timestamp - state.lastUiRenderAt >= CONFIG.uiUpdateIntervalMs) {
    renderUi();
    state.lastUiRenderAt = timestamp;
  }

  if (timestamp - state.lastRecommendationAt >= CONFIG.recommendationsIntervalMs) {
    updateRecommendations();
    state.lastRecommendationAt = timestamp;
  }

  requestAnimationFrame(loop);
}

function updateSimulation(deltaMs) {
  maintainQueue();
  processAdmissions();
  layoutQueue();

  for (const agent of state.agents) {
    if (
      agent.status === "approachingQueue" ||
      agent.status === "moving" ||
      agent.status === "entering" ||
      agent.status === "leaving" ||
      agent.status === "outsideLeaving"
    ) {
      moveAgent(agent, deltaMs);
    } else if (agent.status === "dwelling") {
      updateDwellingAgent(agent, deltaMs);
    }
  }

  updateBleSignals();
}

function getClubOccupancy() {
  return state.agents.filter((agent) => CLUB_STATUSES.has(agent.status)).length;
}

function getActualZoneCount(zoneName) {
  return state.agents.filter((agent) => CLUB_STATUSES.has(agent.status) && agent.currentZone === zoneName).length;
}

function getSelectedAgent() {
  if (state.selectedAgentId === null) {
    return null;
  }

  return state.agents.find((agent) => agent.id === state.selectedAgentId) || null;
}

function getNearestVisibleAgent(point) {
  const wrapperRect = dom.mapWrapper.getBoundingClientRect();
  const clickRadiusInSvg = Math.max(28, (24 / Math.max(1, wrapperRect.width)) * CONFIG.viewBox.width);
  let nearest = null;
  let nearestDistance = Infinity;

  for (const agent of state.agents) {
    if (!VISIBLE_STATUSES.has(agent.status)) {
      continue;
    }

    if (agent.status === "queue" && agent.queuePositionIndex >= CONFIG.visibleQueueLimit) {
      continue;
    }

    const distance = Math.hypot(agent.x - point.x, agent.y - point.y);
    if (distance < nearestDistance && distance <= clickRadiusInSvg) {
      nearest = agent;
      nearestDistance = distance;
    }
  }

  return nearest;
}

function getRouteLabel(agent) {
  if (!agent.route || agent.route.length === 0 || agent.routeIndex >= agent.route.length) {
    return "нет активного маршрута";
  }

  return agent.route
    .slice(agent.routeIndex)
    .map((point) => point.name || `${Math.round(point.x)},${Math.round(point.y)}`)
    .join(" -> ");
}

function clientToSvgPoint(event) {
  const rect = dom.mapWrapper.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * CONFIG.viewBox.width,
    y: ((event.clientY - rect.top) / rect.height) * CONFIG.viewBox.height
  };
}

function formatSignal(value) {
  return Number.isFinite(value) ? `${value} dBm` : "нет сигнала";
}

function formatSignalLine(agent) {
  return BEACON_DISPLAY_ORDER
    .map((name) => `${name}: ${formatSignal(agent.bleSignals[name])}`)
    .join(" · ");
}

function formatTopBeacons(agent) {
  if (!agent.bleTopBeacons || agent.bleTopBeacons.length === 0) {
    return "нет доступных маяков";
  }

  return agent.bleTopBeacons
    .map((beacon) => `${beacon.name} ${beacon.rssi} dBm`)
    .join(", ");
}

function getBeaconShortLabel(name) {
  const labels = {
    entrance: "вход/выход",
    bar: "бар",
    dancefloor: "танцпол",
    veranda: "веранда"
  };

  return labels[name] || name;
}

function getLocationLabel(name) {
  const labels = {
    outsideBle: "вне BLE",
    entrance: "вход/выход",
    bar: "бар",
    dancefloor: "танцпол",
    veranda: "веранда"
  };

  return labels[name] || name;
}

function setText(element, value) {
  element.textContent = Number(value).toLocaleString("ru-RU");
}

function showError(message) {
  dom.errorBanner.hidden = false;
  dom.errorBanner.textContent = message;
}

function randomBetween(min, max) {
  return min + Math.random() * (max - min);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function hashString(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function seededStep(seed) {
  let value = seed >>> 0;
  value ^= value << 13;
  value ^= value >>> 17;
  value ^= value << 5;
  return value >>> 0;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

setupEventListeners();
dom.speedValue.textContent = `${state.speedMultiplier.toFixed(2)}x`;
loadUserIds();
requestAnimationFrame(loop);
