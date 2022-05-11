<template>
    <div id="data-content">
        <div id="mode-select">
            <el-select v-model="matrixType" @change="changeMatrixType">
                <el-option
                    v-for="item in matrixModes"
                    :key="item.value"
                    :label="item.value"
                    :value="item.value">
                </el-option>
            </el-select>
            <el-select v-model="return_mode" @change="changeMode">
                <el-option
                    v-for="item in data_mode"
                    :key="item.value"
                    :label="item.value"
                    :value="item.value">
                </el-option>
            </el-select>
        </div>
        <div id="confusion-matrix-container">
            <confusion-matrix v-if="matrixType==='confusion'" ref="matrix" id="confusion-matrix" @changeMatrix="changeMatrix"
                :showColor="true" :confusionMatrix="confusionMatrix" :returnMode="return_mode"></confusion-matrix>
            <numerical-matrix v-else-if="matrixType!=='confusion'" ref="numerical" @changeMatrix="changeMatrix"
                :numericalMatrix="numericalMatrix" :numericalMatrixType="matrixType" :returnMode="return_mode"></numerical-matrix>
        </div>
    </div>
</template>

<script>
import Vue from 'vue';
import ConfusionMatrix from './ConfusionMatrix.vue';
import NumericalMatrix from './NumericalMatrix.vue';
import axios from 'axios';
import {Select, Option} from 'element-ui';

Vue.use(Select);
Vue.use(Option);

export default {
    components: {ConfusionMatrix, NumericalMatrix},
    name: 'DataView',
    data() {
        return {
            matrixType: 'confusion', // confusion, size
            confusionMatrix: undefined,
            numericalMatrix: undefined,
            return_mode: 'count',
            data_mode: [{
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
            matrixModes: [{
                value: 'confusion',
                label: '混淆矩阵',
            }, {
                value: 'size',
                label: '物体框大小矩阵',
            }],
            query: {},
        };
    },
    methods: {
        setMatrix: function(query) {
            this.confusionMatrix = undefined;
            this.numericalMatrix = undefined;
            // query['return'] = ['count', this.return_mode];
            if (this.return_mode!=='count') {
                query['return'] = ['count', this.return_mode];
            } else {
                query['return'] = ['count'];
            }
            this.query = query;
            const store = this.$store;
            const that = this;
            if (this.matrixType==='confusion') {
                axios.post(store.getters.URL_GET_CONFUSION_MATRIX, that.query===undefined?{}:{query: that.query})
                    .then(function(response) {
                        that.confusionMatrix = response.data;
                    });
            } else {
                if (this.matrixType==='size') {
                    axios.post(store.getters.URL_GET_SIZE_MATRIX, that.query===undefined?{}:{query: that.query})
                        .then(function(response) {
                            const numericalMatrix = response.data;
                            numericalMatrix.counts = numericalMatrix.matrix[0];
                            numericalMatrix.matrix = numericalMatrix.matrix[numericalMatrix.matrix.length-1];
                            for (let i=0; i<numericalMatrix.partitions.length; i++) {
                                numericalMatrix.partitions[i] = Math.round(numericalMatrix.partitions[i]*100)/100;
                            }
                            numericalMatrix.partitions.push('N');
                            that.numericalMatrix = numericalMatrix;
                        });
                }
            }
        },
        changeMatrix: function(matrixType, query) {
            this.query = query;
            this.matrixType = matrixType;
            this.setMatrix(this.query);
        },
        changeMatrixType: function() {
            this.setMatrix({});
        },
        changeMode: function() {
            this.setMatrix(this.query);
        },
    },
    mounted: function() {
        this.setMatrix(this.query);
    },
};
</script>

<style scoped>
#data-content {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

#confusion-matrix-container {
  width: 100%;
  height: 100%;
  display: flex;
  margin: 10px 10px 10px 10px;
}

#mode-select {
    margin: auto;
    padding: 10px;
}
</style>
