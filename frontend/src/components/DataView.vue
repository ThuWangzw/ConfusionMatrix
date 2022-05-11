<template>
    <div id="data-content">
        <div id="mode-select">
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
            <confusion-matrix v-if="matrixType==='confusion'" ref="matrix" id="confusion-matrix" @setMatrix="setMatrix"
                :showColor="true" :confusionMatrix="confusionMatrix"></confusion-matrix>
            <numerical-matrix v-else-if="matrixType==='numerical'" ref="numerical" @setMatrix="setMatrix"
                :numericalMatrix="numericalMatrix" :numericalMatrixType="numericalMatrixType"></numerical-matrix>
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
            matrixType: 'numerical', // confusion, numerical
            numericalMatrixType: 'size',
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
            query: {},
        };
    },
    methods: {
        getMatrix: function(query) {
            const store = this.$store;
            const that = this;
            if (this.matrixType==='confusion') {
                axios.post(store.getters.URL_GET_CONFUSION_MATRIX, query===undefined?{}:{query})
                    .then(function(response) {
                        if (that.return_mode==='count') that.confusionMatrix = [response.data[0]];
                        else that.confusionMatrix = response.data;
                    });
            } else if (this.matrixType==='numerical') {
                if (this.numericalMatrixType==='size') {
                    axios.post(store.getters.URL_GET_SIZE_MATRIX, query===undefined?{}:{query})
                        .then(function(response) {
                            const numericalMatrix = response.data;
                            numericalMatrix.matrix = numericalMatrix.matrix[1];
                            for (let i=0; i<numericalMatrix.partitions.length; i++) {
                                numericalMatrix.partitions[i] = Math.round(numericalMatrix.partitions[i]*100)/100;
                            }
                            numericalMatrix.partitions.push('N');
                            that.numericalMatrix = numericalMatrix;
                        });
                }
            }
        },
        setMatrix: function(matrixType, query, numericalMatrixType) {
            this.confusionMatrix = undefined;
            this.numericalMatrix = undefined;
            this.matrixType = matrixType;
            this.numericalMatrixType = numericalMatrixType;
            query['return'] = ['count', this.return_mode];
            this.query = query;
            this.getMatrix(query);
        },
        changeMode: function(val) {
            this.setMatrix(this.matrixType, this.query, this.numericalMatrixType);
        },
    },
    mounted: function() {
        this.setMatrix(this.matrixType, this.query, this.numericalMatrixType);
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
