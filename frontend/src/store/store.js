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
        URL_GET_BOX_ASPECT_RATIO_DIST: (state) => state.APIBASE + '/api/boxAspectRatioDist',
        URL_GET_BOX_SIZE_DIST: (state) => state.APIBASE + '/api/boxSizeDist',
        URL_GET_IMAGE: (state) => {
            return (boxID, showmode, showAllBox) => state.APIBASE + `/api/image?boxID=${boxID}&show=${showmode}&showall=${showAllBox}`;
        },
        URL_GET_IMAGES: (state) => state.APIBASE + '/api/images',
        URL_GET_IMAGES_IN_MATRIX_CELL: (state) => state.APIBASE+'/api/imagesInCell',
        URL_GET_GRID: (state) => state.APIBASE + '/api/grid',
    },
});
