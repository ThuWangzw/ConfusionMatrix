<template>
    <div id="data-content">
        <div id="left-widgets">
            <div class="toolbox">
                <div id="encoding-select">
                    <span class="select-label">Matrix Encoding</span>
                    <el-select v-model="returnMode" @change="changeMode" size="mini">
                        <el-option
                            v-for="item in dataMode"
                            :key="item.value"
                            :label="item.value"
                            :value="item.value">
                        </el-option>
                    </el-select>
                </div>
                <i v-if="gettingMatrix||gettingSizeBarchart||gettingAspectRatioBarchart" class="el-icon-loading"></i>
            </div>

            <div id="scented-barcharts">
                <scented-barchart ref="label-size-hist"
                    :allData="labelSizeAll" :title="'gt_box_size'" queryKey="label_size" :selectData="labelSizeSelect" :xSplit="sizeSplit"
                    @hoverBarchart="hoverBarchart"></scented-barchart>
                <scented-barchart ref="predict-size-hist"
                    :allData="predictSizeAll" :title="'pred_box_size'" queryKey="predict_size" :selectData="predictSizeSelect" :xSplit="sizeSplit"
                    @hoverBarchart="hoverBarchart"></scented-barchart>
                <scented-barchart ref="label-aspect-ratio-hist"
                    :allData="labelAspectRatioAll" :title="'gt_box_aspect_ratio'" queryKey="label_aspect_ratio"
                    :selectData="labelAspectRatioSelect" :xSplit="aspectRatioSplit"
                    @hoverBarchart="hoverBarchart"></scented-barchart>
                <scented-barchart ref="predict-aspect_ratio-hist"
                    :allData="predictAspectRatioAll" :title="'pred_box_aspect_ratio'" queryKey="predict_aspect_ratio"
                    :selectData="predictAspectRatioSelect" :xSplit="aspectRatioSplit"
                    @hoverBarchart="hoverBarchart"></scented-barchart>
            </div>
        </div>
        <div id="matrices-container">
            <div id="confusion-matrix-container">
                <confusion-matrix ref="matrix" @hoverConfusion="hoverConfusion"
                    :showColor="true" :confusionMatrix="confusionMatrix" :returnMode="returnMode"></confusion-matrix>
            </div>
        </div>
    </div>
</template>

<script>
import Vue from 'vue';
import ConfusionMatrix from './ConfusionMatrix.vue';
import ScentedBarchart from './ScentedBarchart.vue';
import axios from 'axios';
import {Select, Option, Icon} from 'element-ui';

Vue.use(Select);
Vue.use(Option);
Vue.use(Icon);

export default {
    components: {ConfusionMatrix, ScentedBarchart},
    name: 'DataView',
    data() {
        return {
            confusionMatrix: undefined,
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
        setConfusionMatrix: function(query) {
            this.gettingMatrix = true;
            // this.confusionMatrix = undefined;
            if (query===undefined) {
                query = {};
            }
            if (this.returnMode!=='count') {
                query['return'] = ['count', this.returnMode];
            } else {
                query['return'] = ['count'];
            }
            // this.query = query;
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
        changeMode: function() {
            this.setConfusionMatrix({});
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
            for (let i = 0; i < 10; ++i) {
                tmp1.push(0);
                tmp2.push(0);
                tmp3.push(0);
                tmp4.push(0);
            }
            for (const i of labelClasses) {
                for (const j of predictClasses) {
                    for (let k = 0; k < 10; ++k) {
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
            this.gettingSizeBarchart = true;
            this.gettingAspectRatioBarchart = true;
            this.query = {...this.query, ...query};
            this.setConfusionMatrix(this.query);
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
    font-size: 15px;
    margin-right: 15px;
}

#encoding-select>.el-select {
    width: 100px;
}
.toolbox {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

#data-content {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: row;
}

#matrices-container {
  width: 80%;
  height: 100%;
  display: flex;
  margin: 10px 10px 10px 10px;
}

#confusion-matrix-container {
    width: 100%;
    height: 80%;
}


#left-widgets {
    margin: 0;
    padding: 10px;
    width: 20%;
}

#scented-barcharts {
    width: 100%;
    height: 20%;
    margin: auto;
}

svg {
    margin: 10px;
}

</style>
