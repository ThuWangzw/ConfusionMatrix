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
                    <div class="mode-select">
                        <span class="select-label">Show Direction</span>
                        <el-button id="direction-button" size="mini" @click="changeShowDirection">{{showDirection}}</el-button>
                    </div>
                </div>
            </div>

            <div id="barcharts-container">
                <div class="toolbar-title">Filters</div>
                <div id="scented-barcharts">
                    <scented-barchart ref="label-size-hist" :barNum="barNum"
                        :allData="labelSizeAll" :title="'GT_size'" queryKey="label_size"
                        :selectData="labelSizeSelect" :xSplit="sizeSplit" :displayMode="displayMode"
                        @hoverBarchart="hoverBarchart"></scented-barchart>
                    <scented-barchart ref="predict-size-hist" :barNum="barNum"
                        :allData="predictSizeAll" :title="'PR_size'" queryKey="predict_size"
                        :selectData="predictSizeSelect" :xSplit="sizeSplit" :displayMode="displayMode"
                        @hoverBarchart="hoverBarchart"></scented-barchart>
                    <scented-barchart ref="label-aspect-ratio-hist" :barNum="barNum"
                        :allData="labelAspectRatioAll" :title="'GT_AR'" queryKey="label_aspect_ratio"
                        :selectData="labelAspectRatioSelect" :xSplit="aspectRatioSplit" :displayMode="displayMode"
                        @hoverBarchart="hoverBarchart"></scented-barchart>
                    <scented-barchart ref="predict-aspect_ratio-hist" :barNum="barNum"
                        :allData="predictAspectRatioAll" :title="'PR_AR'" queryKey="predict_aspect_ratio"
                        :selectData="predictAspectRatioSelect" :xSplit="aspectRatioSplit" :displayMode="displayMode"
                        @hoverBarchart="hoverBarchart"></scented-barchart>
                </div>
            </div>

            <div id="selection-container">
                <div class="toolbar-title">Selections</div>
                <div id="selections">
                    selections
                </div>
            </div>

        </div>
        <div id="matrices-container">
            <div class="toolbar-title">Matrix</div>
            <div id="confusion-matrix-container">
                <confusion-matrix ref="matrix" @hoverConfusion="hoverConfusion" :showDirection="showDirection"
                    :confusionMatrix="confusionMatrix" :returnMode="returnMode"></confusion-matrix>
            </div>
        </div>
    </div>
</template>

<script>
import Vue from 'vue';
import ConfusionMatrix from './ConfusionMatrix.vue';
import ScentedBarchart from './ScentedBarchart.vue';
import axios from 'axios';
import {Select, Option, Icon, Button} from 'element-ui';

Vue.use(Select);
Vue.use(Option);
Vue.use(Icon);
Vue.use(Button);

export default {
    components: {ConfusionMatrix, ScentedBarchart},
    name: 'DataView',
    data() {
        return {
            displayMode: 'log',
            barNum: 25,
            confusionMatrix: undefined,
            showDirection: false,
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
            sizeSplit: [],
            labelSizeAll: [],
            predictSizeAll: [],
            labelSizeConfusion: undefined,
            predictSizeConfusion: undefined,
            labelSizeSelect: [],
            predictSizeSelect: [],
            labelSizeSelectBuffer: [],
            predictSizeSelectBuffer: [],
            aspectRatioSplit: [],
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
        };
    },
    methods: {
        changeDisplayMode: function() {
            if (this.displayMode === 'log') this.displayMode = 'linear';
            else this.displayMode = 'log';
            document.getElementById('log-linear-button').blur();
        },
        changeShowDirection: function() {
            this.showDirection = !this.showDirection;
            document.getElementById('direction-button').blur();
        },
        setConfusionMatrix: function(query) {
            this.gettingMatrix = true;
            // this.confusionMatrix = undefined;
            if (query===undefined) {
                query = {};
            }
            const returnList = ['count'];
            if (this.returnMode!=='count') returnList.push(this.returnMode);
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
                    that.sizeSplit = response.data.sizeSplit;
                    that.labelSizeAll = response.data.labelSizeAll;
                    that.predictSizeAll = response.data.predictSizeAll;
                    that.labelSizeSelectBuffer = that.labelSizeAll;
                    that.predictSizeSelectBuffer = that.predictSizeAll;
                    that.labelSizeConfusion = response.data.labelSizeConfusion;
                    that.predictSizeConfusion = response.data.predictSizeConfusion;
                    that.labelSizeSelect = response.data.labelSizeAll;
                    that.predictSizeSelect = response.data.predictSizeAll;
                    // that.sizeSplit.unshift(0);
                    // that.sizeSplit.push(1);
                    that.gettingSizeBarchart = false;
                });
        },
        setBoxAspectRatioInfo: function(query) {
            this.gettingAspectRatioBarchart = true;
            const store = this.$store;
            const that = this;
            axios.post(store.getters.URL_GET_BOX_ASPECT_RATIO_DIST, query===undefined?{}:{query: query})
                .then(function(response) {
                    that.aspectRatioSplit = response.data.aspectRatioSplit;
                    that.labelAspectRatioAll = response.data.labelAspectRatioAll;
                    that.predictAspectRatioAll = response.data.predictAspectRatioAll;
                    that.labelAspectRatioSelectBuffer = that.labelAspectRatioAll;
                    that.predictAspectRatioSelectBuffer = that.predictAspectRatioAll;
                    that.labelAspectRatioConfusion = response.data.labelAspectRatioConfusion;
                    that.predictAspectRatioConfusion = response.data.predictAspectRatioConfusion;
                    that.labelAspectRatioSelect = response.data.labelAspectRatioAll;
                    that.predictAspectRatioSelect = response.data.predictAspectRatioAll;
                    // that.aspectRatioSplit.unshift(0);
                    // that.aspectRatioSplit.push(1);
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
            axios.post(store.getters.URL_GET_BOX_SIZE_DIST, query===undefined?{}:{query: this.query})
                .then(function(response) {
                    that.labelSizeSelect = response.data.labelSizeAll;
                    that.predictSizeSelect = response.data.predictSizeAll;
                    that.labelSizeConfusion = response.data.labelSizeConfusion;
                    that.predictSizeConfusion = response.data.predictSizeConfusion;
                    that.labelSizeSelectBuffer = response.data.labelSizeAll;
                    that.predictSizeSelectBuffer = response.data.predictSizeAll;
                    that.gettingSizeBarchart = false;
                });
            axios.post(store.getters.URL_GET_BOX_ASPECT_RATIO_DIST, query===undefined?{}:{query: this.query})
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
    },
    mounted: function() {
        this.setBoxSizeInfo();
        this.setBoxAspectRatioInfo();
        this.setConfusionMatrix();
    },
};
</script>

<style scoped>
.select-label {
    font-family: Comic Sans MS;
    font-weight: normal;
    font-size: 10px;
    /* margin-right: 5px; */
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
    width: 100%;
    height: 100%;
    display: flex;
    margin: 0 0 0 10px;
    flex-direction: column;
}

#confusion-matrix-container {
    /* width: 100%; */
    height: 100%;
    border: 1px solid #c1c1c1;
    border-radius: 5px;
}


#left-widgets {
    padding: 2px;
    width: 18%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

#scented-barcharts>svg {
    width: 100%;
    height: 35px;
}

#scented-barcharts {
    border: 1px solid #c1c1c1;
    border-radius: 5px;
}

</style>
