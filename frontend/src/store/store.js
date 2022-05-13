import Vue from 'vue';
import Vuex from 'vuex';
Vue.use(Vuex);

export default new Vuex.Store({
    state: {
        APIBASE: BACKEND_BASE_URL,
        labelHierarchy: undefined,
        labelnames: [],
        colors: {},
        hierarchyColors: undefined,
    },
    mutations: {
        setMetadata(state, metadata) {
            state.labelHierarchy = metadata.hierarchy;
            state.labelnames = metadata.names;
        },
        setColors(state, colors) {
            state.colors = colors;
        },
        setHierarchyColors(state, hierarchyColors) {
            state.hierarchyColors = hierarchyColors;
        },
    },
    getters: {
        confusionMatrix: (state) => state.confusionMatrix,
        labelHierarchy: (state) => state.labelHierarchy,
        labelnames: (state) => state.labelnames,
        colors: (state) => state.colors,
        hierarchyColors: (state) => state.hierarchyColors,
        URL_GET_METADATA: (state) => state.APIBASE + '/api/metadata',
        URL_GET_CONFUSION_MATRIX: (state) => state.APIBASE + '/api/confusionMatrix',
        URL_GET_SIZE_MATRIX: (state) => state.APIBASE + '/api/sizeMatrix',
        URL_GET_BOX_SIZE_INFO: (state) => state.APIBASE + '/api/boxSizeInfo',
        URL_GET_BOX_SIZE_DIST: (state) => state.APIBASE + '/api/boxSizeDist',
    },
});
