import Vue from 'vue';
import Vuex from 'vuex';
Vue.use(Vuex);

export default new Vuex.Store({
    state: {
        APIBASE: BACKEND_BASE_URL,
        confusionMatrix: [],
        labelHierarchy: [],
        labelnames: [],
        colors: {},
        hierarchyColors: {},
        confusionCellID: null, // which cell clicked ({labels, preds}) in dataview
        shownClass: [], // visible classes in gridlayout, used for expand of confusion matrix
    },
    mutations: {
        setConfusionMatrix(state, confusionMatrix) {
            state.confusionMatrix = confusionMatrix.matrix;
            state.labelHierarchy = confusionMatrix.hierarchy;
            state.labelnames = confusionMatrix.names;
        },
        setColors(state, colors) {
            state.colors = colors;
        },
        setConfusionCellID(state, confusionCellID) {
            state.confusionCellID = confusionCellID;
        },
        setConfusionCellID(state, confusionCellID) {
            state.confusionCellID = confusionCellID;
        },
        setHierarchyColors(state, hierarchyColors) {
            state.hierarchyColors = hierarchyColors;
        },
        setShownClass(state, shownClass) {
            state.shownClass = shownClass;
        },
    },
    getters: {
        confusionCellID: (state) => state.confusionCellID,
        confusionMatrix: (state) => state.confusionMatrix,
        labelHierarchy: (state) => state.labelHierarchy,
        labelnames: (state) => state.labelnames,
        colors: (state) => state.colors,
        hierarchyColors: (state) => state.hierarchyColors,
        shownClass: (state) => state.shownClass,
        URL_GET_CONFUSION_MATRIX: (state) => state.APIBASE + '/api/confusionMatrix',
    },
});
