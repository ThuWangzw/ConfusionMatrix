<template>
    <div id="data-content">
        <div id="left-widgets">
            <el-select v-model="returnMode" @change="changeMode">
                <el-option
                    v-for="item in dataMode"
                    :key="item.value"
                    :label="item.value"
                    :value="item.value">
                </el-option>
            </el-select>
            <div id="scented-barcharts">
                <scented-barchart ref="label-hist"
                    :allData="labelSizeAll" :title="'gt_box_size'" :selectData="labelSizeSelect"></scented-barchart>
                <scented-barchart ref="predict-hist"
                    :allData="predictSizeAll" :title="'pred_box_size'" :selectData="predictSizeSelect"></scented-barchart>
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
import {Select, Option} from 'element-ui';

Vue.use(Select);
Vue.use(Option);

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
            labelSizeAll: [],
            predictSizeAll: [],
            labelSizeConfusion: undefined,
            predictSizeConfusion: undefined,
            labelSizeSelect: [],
            predictSizeSelect: [],
        };
    },
    methods: {
        setConfusionMatrix: function(query) {
            // this.confusionMatrix = undefined;
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
                });
        },
        setBoxSizeInfo: function() {
            const store = this.$store;
            const that = this;
            axios.post(store.getters.URL_GET_BOX_SIZE_DIST, {})
                .then(function(response) {
                    that.labelSizeAll = response.data.labelSizeAll;
                    that.predictSizeAll = response.data.predictSizeAll;
                    that.labelSizeConfusion = response.data.labelSizeConfusion;
                    that.predictSizeConfusion = response.data.predictSizeConfusion;
                    that.labelSizeSelect = response.data.labelSizeAll;
                    that.predictSizeSelect = response.data.predictSizeAll;
                });
        },
        changeMode: function() {
            this.setConfusionMatrix({});
        },
        hoverConfusion: function(labelClasses, predictClasses) {
            if (labelClasses === undefined) {
                this.labelSizeSelect = this.labelSizeAll;
                this.predictSizeSelect = this.predictSizeAll;
                return;
            }
            const tmp1 = [];
            const tmp2 = [];
            for (let i = 0; i < 10; ++i) {
                tmp1.push(0);
                tmp2.push(0);
            }
            for (const i of labelClasses) {
                for (const j of predictClasses) {
                    for (let k = 0; k < 10; ++k) {
                        tmp1[k] += this.labelSizeConfusion[i][j][k];
                        tmp2[k] += this.predictSizeConfusion[i][j][k];
                    }
                }
            }
            this.labelSizeSelect = tmp1;
            this.predictSizeSelect = tmp2;
        },
    },
    mounted: function() {
        this.setBoxSizeInfo();
        this.setConfusionMatrix({});
    },
};
</script>

<style scoped>
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
