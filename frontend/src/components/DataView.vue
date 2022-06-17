<template>
    <div id="data-content">
        <div id="left-widgets">
            <div id="toolbox-container">
                <div class="toolbar-title">Settings</div>
                <div class="toolbox">
                    <div class="mode-select">
                        <span class="select-label">Matrix Encoding</span>
                        <el-select v-model="returnMode" @change="changeDataMode" size="mini">
                            <el-option
                                v-for="item in dataMode"
                                :key="item.value"
                                :label="item.value"
                                :value="item.value">
                            </el-option>
                        </el-select>
                        <i v-if="gettingMatrix||gettingSizeBarchart||gettingAspectRatioBarchart" class="el-icon-loading"></i>
                    </div>
                    <div class="mode-select">
                        <span class="select-label">Display Mode</span>
                        <el-button id="log-linear-button" size="mini" @click="changeDisplayMode">{{displayMode}}</el-button>
                    </div>
                </div>
            </div>

            <div id="barcharts-container">
                <div class="toolbar-title">Filters</div>
                <div id="scented-barcharts">
                    <scented-barchart ref="label-size-hist" :barNum="barNum" :dataRangeAll="labelSizeRange"
                        :allData="labelSizeAll" :title="'GT_size'" queryKey="label_size" :overallDist="labelSizeOverallDist"
                        :selectData="labelSizeSelect" :xSplit="labelSizeSplit" :displayMode="displayMode"
                        @hoverBarchart="hoverBarchart" @selectRange="selectRange"></scented-barchart>
                    <scented-barchart ref="predict-size-hist" :barNum="barNum" :dataRangeAll="predictSizeRange"
                        :allData="predictSizeAll" :title="'PR_size'" queryKey="predict_size" :overallDist="predictSizeOverallDist"
                        :selectData="predictSizeSelect" :xSplit="predictSizeSplit" :displayMode="displayMode"
                        @hoverBarchart="hoverBarchart" @selectRange="selectRange"></scented-barchart>
                    <scented-barchart ref="label-aspect-ratio-hist" :barNum="barNum" :dataRangeAll="labelAspectRatioRange"
                        :allData="labelAspectRatioAll" :title="'GT_AR'" queryKey="label_aspect_ratio" :overallDist="labelAspectRatioOverallDist"
                        :selectData="labelAspectRatioSelect" :xSplit="labelAspectRatioSplit" :displayMode="displayMode"
                        @hoverBarchart="hoverBarchart" @selectRange="selectRange"></scented-barchart>
                    <scented-barchart ref="predict-aspect_ratio-hist" :barNum="barNum" :dataRangeAll="predictAspectRatioRange"
                        :allData="predictAspectRatioAll" :title="'PR_AR'" queryKey="predict_aspect_ratio" :overallDist="predictAspectRatioOverallDist"
                        :selectData="predictAspectRatioSelect" :xSplit="predictAspectRatioSplit" :displayMode="displayMode"
                        @hoverBarchart="hoverBarchart" @selectRange="selectRange"></scented-barchart>
                </div>
            </div>

            <div id="selection-container">
                <div class="toolbar-title">Colors</div>
                <div id="selections">
                    <selection :confusionMatrix="confusionMatrix"></selection>
                </div>
            </div>

        </div>
        <div id="matrices-container">
            <div class="toolbar-title" id="grid-toolbar">
                <span>Confusion Matrix</span>
                <div id="grid-icons">
                    <img id="matrix-normal-icon" class="grid-icon" src="/static/images/square.svg" @click="changeShowNormal">
                    <img id="matrix-direction-icon" class="grid-icon" src="/static/images/directions.svg" @click="changeShowDirection">
                    <img id="matrix-size-comparison-icon" class="grid-icon" src="/static/images/circle.png" @click="changeShowSizeComparison">
                </div>
            </div>
            <div id="confusion-matrix-container">
                <confusion-matrix ref="matrix" @hoverConfusion="hoverConfusion" :showMode="showMode" @clickCell="clickConfusionCell"
                    :confusionMatrix="confusionMatrix" :returnMode="returnMode"></confusion-matrix>
            </div>
        </div>
        <div id="grid-view-container">
            <div class="toolbar-title" id="grid-toolbar">
                <span>Images</span>
                <div id="grid-icons">
                    <img id="grid-zoomin-icon" class="grid-icon" src="/static/images/zoomin.svg" @click="initGridLayoutLasso">
                    <img id="grid-home-icon" class="grid-icon" src="/static/images/home.png" @click="gridLayoutZoomin()">
                </div>
            </div>
            <div id="grid-layout-container">
                <grid-layout ref="grid"></grid-layout>
            </div>
        </div>
    </div>
</template>

<script>
import Vue from 'vue';
import ConfusionMatrix from './ConfusionMatrix.vue';
import ScentedBarchart from './ScentedBarchart.vue';
import Selection from './Selection.vue';
import GridLayout from './GridLayout.vue';
import axios from 'axios';
import {Select, Option, Icon, Button} from 'element-ui';

Vue.use(Select);
Vue.use(Option);
Vue.use(Icon);
Vue.use(Button);

export default {
    components: {ConfusionMatrix, ScentedBarchart, GridLayout, Selection},
    name: 'DataView',
    data() {
        return {
            displayMode: 'linear',
            barNum: 25,
            confusionMatrix: undefined,
            showMode: 'normal',
            returnMode: 'count',
            dataMode: [{
                value: 'count',
                label: '数量',
            }, {
                value: 'avg_label_size',
                label: '真实物体框平均大小',
            }, {
                value: 'avg_predict_size',
                label: '预测物体框平均大小',
            }, {
                value: 'avg_iou',
                label: 'iou均值',
            }, {
                value: 'avg_acc',
                label: '准确率均值',
            }, {
                value: 'avg_label_aspect_ratio',
                label: '真实物体框纵横比均值',
            }, {
                value: 'avg_predict_aspect_ratio',
                label: '预测物体框纵横比均值',
            }],
            query: {},
            labelSizeSplit: [],
            predictSizeSplit: [],
            labelSizeAll: [],
            predictSizeAll: [],
            labelSizeConfusion: undefined,
            predictSizeConfusion: undefined,
            labelSizeSelect: [],
            predictSizeSelect: [],
            labelSizeSelectBuffer: [],
            predictSizeSelectBuffer: [],
            labelAspectRatioSplit: [],
            predictAspectRatioSplit: [],
            labelAspectRatioAll: [],
            predictAspectRatioAll: [],
            labelAspectRatioConfusion: undefined,
            predictAspectRatioConfusion: undefined,
            labelAspectRatioSelect: [],
            predictAspectRatioSelect: [],
            labelAspectRatioSelectBuffer: [],
            predictAspectRatioSelectBuffer: [],
            gettingMatrix: true,
            gettingSizeBarchart: true,
            gettingAspectRatioBarchart: true,
            labelSizeRange: undefined,
            predictSizeRange: undefined,
            labelAspectRatioRange: undefined,
            predictAspectRatioRange: undefined,
            labelSizeShow: undefined,
            predictSizeShow: undefined,
            labelAspectRatioShow: undefined,
            predictAspectRatioShow: undefined,
            labelSizeOverallDist: undefined,
            predictSizeOverallDist: undefined,
            labelAspectRatioOverallDist: undefined,
            predictAspectRatioOverallDist: undefined,
        };
    },
    methods: {
        changeDisplayMode: function() {
            if (this.displayMode === 'log') this.displayMode = 'linear';
            else this.displayMode = 'log';
            document.getElementById('log-linear-button').blur();
        },
        changeShowDirection: function() {
            this.showMode = 'direction';
        },
        changeShowSizeComparison: function() {
            this.showMode = 'sizeComparison';
        },
        changeShowNormal: function() {
            this.showMode = 'normal';
        },
        setConfusionMatrix: function(query) {
            this.gettingMatrix = true;
            // this.confusionMatrix = undefined;
            if (query===undefined) {
                query = {};
            }
            const returnList = ['count'];
            if (this.returnMode!=='count') returnList.push(this.returnMode);
            returnList.push('size_comparison');
            returnList.push('direction');
            query['return'] = returnList;
            const store = this.$store;
            const that = this;
            axios.post(store.getters.URL_GET_CONFUSION_MATRIX, query===undefined?{}:{query: query})
                .then(function(response) {
                    that.confusionMatrix = response.data;
                    that.gettingMatrix = false;
                });
        },
        setBoxSizeInfo: function(query) {
            this.gettingSizeBarchart = true;
            const store = this.$store;
            const that = this;
            axios.post(store.getters.URL_GET_BOX_SIZE_DIST, query===undefined?{}:{query: query})
                .then(function(response) {
                    that.labelSizeSplit = response.data.labelSplit;
                    that.predictSizeSplit = response.data.predictSplit;
                    that.labelSizeRange = [that.labelSizeSplit[0], that.labelSizeSplit[that.labelSizeSplit.length-1]];
                    that.labelSizeShow = that.labelSizeRange;
                    that.predictSizeRange = [that.predictSizeSplit[0], that.predictSizeSplit[that.predictSizeSplit.length-1]];
                    that.predictSizeShow = that.predictSizeRange;
                    that.labelSizeAll = response.data.labelSizeAll;
                    that.predictSizeAll = response.data.predictSizeAll;
                    that.labelSizeSelectBuffer = that.labelSizeAll;
                    that.predictSizeSelectBuffer = that.predictSizeAll;
                    that.labelSizeConfusion = response.data.labelSizeConfusion;
                    that.predictSizeConfusion = response.data.predictSizeConfusion;
                    that.labelSizeSelect = response.data.labelSizeAll;
                    that.predictSizeSelect = response.data.predictSizeAll;
                    that.gettingSizeBarchart = false;
                });
        },
        setBoxAspectRatioInfo: function(query) {
            this.gettingAspectRatioBarchart = true;
            const store = this.$store;
            const that = this;
            axios.post(store.getters.URL_GET_BOX_ASPECT_RATIO_DIST, query===undefined?{}:{query: query})
                .then(function(response) {
                    that.labelAspectRatioSplit = response.data.labelSplit;
                    that.predictAspectRatioSplit = response.data.predictSplit;
                    that.labelAspectRatioRange = [that.labelAspectRatioSplit[0],
                        that.labelAspectRatioSplit[that.labelAspectRatioSplit.length-1]];
                    that.labelAspectRatioShow = that.labelAspectRatioRange;
                    that.predictAspectRatioRange = [that.predictAspectRatioSplit[0],
                        that.predictAspectRatioSplit[that.predictAspectRatioSplit.length-1]];
                    that.predictAspectRatioShow = that.predictAspectRatioRange;
                    that.labelAspectRatioAll = response.data.labelAspectRatioAll;
                    that.predictAspectRatioAll = response.data.predictAspectRatioAll;
                    that.labelAspectRatioSelectBuffer = that.labelAspectRatioAll;
                    that.predictAspectRatioSelectBuffer = that.predictAspectRatioAll;
                    that.labelAspectRatioConfusion = response.data.labelAspectRatioConfusion;
                    that.predictAspectRatioConfusion = response.data.predictAspectRatioConfusion;
                    that.labelAspectRatioSelect = response.data.labelAspectRatioAll;
                    that.predictAspectRatioSelect = response.data.predictAspectRatioAll;
                    that.gettingAspectRatioBarchart = false;
                });
        },
        changeDataMode: function() {
            this.setConfusionMatrix();
        },
        hoverConfusion: function(labelClasses, predictClasses) {
            if (labelClasses === undefined) {
                this.labelSizeSelect = this.labelSizeSelectBuffer;
                this.predictSizeSelect = this.predictSizeSelectBuffer;
                this.labelAspectRatioSelect = this.labelAspectRatioSelectBuffer;
                this.predictAspectRatioSelect = this.predictAspectRatioSelectBuffer;
                return;
            }
            const tmp1 = [];
            const tmp2 = [];
            const tmp3 = [];
            const tmp4 = [];
            for (let i = 0; i < this.barNum; ++i) {
                tmp1.push(0);
                tmp2.push(0);
                tmp3.push(0);
                tmp4.push(0);
            }
            for (const i of labelClasses) {
                for (const j of predictClasses) {
                    for (let k = 0; k < this.barNum; ++k) {
                        tmp1[k] += this.labelSizeConfusion[i][j][k];
                        tmp2[k] += this.predictSizeConfusion[i][j][k];
                        tmp3[k] += this.labelAspectRatioConfusion[i][j][k];
                        tmp4[k] += this.predictAspectRatioConfusion[i][j][k];
                    }
                }
            }
            this.labelSizeSelect = tmp1;
            this.predictSizeSelect = tmp2;
            this.labelAspectRatioSelect = tmp3;
            this.predictAspectRatioSelect = tmp4;
        },
        hoverBarchart: function(query) {
            if (query===undefined) {
                query = {};
            }
            this.query = {...this.query, ...query};
            this.setConfusionMatrix(this.query);
            this.gettingSizeBarchart = true;
            this.gettingAspectRatioBarchart = true;
            const store = this.$store;
            const that = this;
            axios.post(store.getters.URL_GET_BOX_SIZE_DIST, {query: {...this.query,
                'label_range': that.labelSizeShow, 'predict_range': that.predictSizeShow}})
                .then(function(response) {
                    that.labelSizeSelect = response.data.labelSizeAll;
                    that.predictSizeSelect = response.data.predictSizeAll;
                    that.labelSizeConfusion = response.data.labelSizeConfusion;
                    that.predictSizeConfusion = response.data.predictSizeConfusion;
                    that.labelSizeSelectBuffer = response.data.labelSizeAll;
                    that.predictSizeSelectBuffer = response.data.predictSizeAll;
                    that.gettingSizeBarchart = false;
                });
            axios.post(store.getters.URL_GET_BOX_ASPECT_RATIO_DIST, {query: {...this.query,
                'label_range': that.labelAspectRatioShow, 'predict_range': that.predictAspectRatioShow}})
                .then(function(response) {
                    that.labelAspectRatioSelect = response.data.labelAspectRatioAll;
                    that.predictAspectRatioSelect = response.data.predictAspectRatioAll;
                    that.labelAspectRatioConfusion = response.data.labelAspectRatioConfusion;
                    that.predictAspectRatioConfusion = response.data.predictAspectRatioConfusion;
                    that.labelAspectRatioSelectBuffer = response.data.labelAspectRatioAll;
                    that.predictAspectRatioSelectBuffer = response.data.predictAspectRatioAll;
                    that.gettingAspectRatioBarchart = false;
                });
        },
        selectRange: function(queryKey, rangeShow) {
            const store = this.$store;
            const that = this;
            const query = this.query;
            query['query_key'] = queryKey;
            query['range'] = rangeShow;
            if (queryKey === 'label_size') {
                that.labelSizeShow = rangeShow;
                this.gettingSizeBarchart = true;
                axios.post(store.getters.URL_GET_ZOOM_IN_DIST, {query: query})
                    .then(function(response) {
                        that.labelSizeAll = response.data.allDist;
                        that.labelSizeSelect = response.data.selectDist;
                        that.labelSizeConfusion = response.data.confusion;
                        that.labelSizeSplit = response.data.split;
                        that.labelSizeSelectBuffer = that.labelSizeSelect;
                        that.gettingSizeBarchart = false;
                    });
            } else if (queryKey == 'label_aspect_ratio') {
                that.labelAspectRatioShow = rangeShow;
                this.gettingAspectRatioBarchart = true;
                axios.post(store.getters.URL_GET_ZOOM_IN_DIST, {query: query})
                    .then(function(response) {
                        that.labelAspectRatioAll = response.data.allDist;
                        that.labelAspectRatioSelect = response.data.selectDist;
                        that.labelAspectRatioConfusion = response.data.confusion;
                        that.labelAspectRatioSplit = response.data.split;
                        that.labelAspectRatioSelectBuffer = that.labelAspectRatioSelect;
                        that.gettingAspectRatioBarchart = false;
                    });
            } else if (queryKey == 'predict_size') {
                that.predictSizeShow = rangeShow;
                this.gettingSizeBarchart = true;
                axios.post(store.getters.URL_GET_ZOOM_IN_DIST, {query: query})
                    .then(function(response) {
                        that.predictSizeAll = response.data.allDist;
                        that.predictSizeSelect = response.data.selectDist;
                        that.predictSizeConfusion = response.data.confusion;
                        that.predictSizeSplit = response.data.split;
                        that.predictSizeSelectBuffer = that.predictSizeSelect;
                        that.gettingSizeBarchart = false;
                    });
            } else if (queryKey == 'predict_aspect_ratio') {
                that.predictAspectRatioShow = rangeShow;
                this.gettingAspectRatioBarchart = true;
                axios.post(store.getters.URL_GET_ZOOM_IN_DIST, {query: query})
                    .then(function(response) {
                        that.predictAspectRatioAll = response.data.allDist;
                        that.predictAspectRatioSelect = response.data.selectDist;
                        that.predictAspectRatioConfusion = response.data.confusion;
                        that.predictAspectRatioSplit = response.data.split;
                        that.predictAspectRatioSelectBuffer = that.predictAspectRatioSelect;
                        that.gettingAspectRatioBarchart = false;
                    });
            } else {
                console.log('query_key error: ' + String(rangeShow));
            }
        },
        setOverallDist: function() {
            const store = this.$store;
            const that = this;
            axios.post(store.getters.URL_GET_OVERALL_DIST, {})
                .then(function(response) {
                    that.labelSizeOverallDist = response.data.labelSize;
                    that.predictSizeOverallDist = response.data.predictSize;
                    that.labelAspectRatioOverallDist = response.data.labelAspectRatio;
                    that.predictAspectRatioOverallDist = response.data.predictAspectRatio;
                });
        },
        clickConfusionCell: function(d) {
            const store = this.$store;
            const that = this;
            axios.post(store.getters.URL_GET_IMAGES_IN_MATRIX_CELL, {
                labels: d.rowNode.leafs,
                preds: d.colNode.leafs,
            }).then(function(response) {
                const images = response.data;
                if (images.length>0) {
                    console.log(images);
                    that.$refs.grid.showBottomNodes(images);
                } else {
                    console.log('no images');
                }
            });
        },
        initGridLayoutLasso: function() {
            this.$refs.grid.initlasso();
        },
        gridLayoutZoomin: function() {
            this.$refs.grid.zoomin();
        },
    },
    mounted: function() {
        this.setBoxSizeInfo();
        this.setBoxAspectRatioInfo();
        this.setConfusionMatrix();
        this.setOverallDist();
    },
};
</script>

<style scoped>
.select-label {
    font-family: Comic Sans MS;
    font-weight: normal;
    font-size: 10px;
    display: block;
    float: left;
    width: 100px;
    line-height: 28px;
}

.mode-select>.el-select {
    width: 100px;
}
.toolbox {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: left;
    flex-direction: column;
    border: 1px solid #c1c1c1;
    border-radius: 5px;
}

.mode-select {
    margin: 2px;
}

#data-content {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: row;
}

#toolbox-container {
    display: flex;
    flex-direction: column;
}

#barcharts-container {
    margin: 5px 0 0 0;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}

#selection-container {
    margin: 5px 0 0 0;
    display: flex;
    flex-direction: column;
    flex: 100 1 auto;
}

#selections {
    height: 100%;
    border: 1px solid #c1c1c1;
    border-radius: 5px;
    flex: 10 1 auto;
}

#matrices-container {
    padding: 2px;
    width: 43%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

#confusion-matrix-container {
    /* width: 100%; */
    height: 100%;
    border: 1px solid #c1c1c1;
    border-radius: 5px;
}

#grid-view-container {
    padding: 2px;
    width: 43%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

#grid-layout-container {
    /* width: 100%; */
    height: 100%;
    border: 1px solid #c1c1c1;
    border-radius: 5px;
}


#left-widgets {
    padding: 2px;
    width: 14%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

#scented-barcharts>svg {
    width: 100%;
    height: 45px;
}

#scented-barcharts {
    border: 1px solid #c1c1c1;
    border-radius: 5px;
}


#grid-icons {
    display: flex;
    justify-content: flex-start;
    align-items: center;
    margin: 0 0 0 25px;
    flex-shrink: 1;
    /* align-self: flex-start; */
}

.grid-icon {
    width: 15px;
    height: 15px;
    margin: 0 5px 0 5px;
    cursor: pointer;
}

#grid-toolbar {
    display: flex;
    flex-direction: row;
    justify-content: flex-start;
    align-items: center;
}
</style>
